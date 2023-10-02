import argparse
import logging
import numpy as np
import os.path as osp
import pandas as pd
import re
from tools import run_model
from shutil import move

# local imports
from data_mapper import MapperHandler, read_input_folder, put_data
from reporter import AdditionalInfoReporter, LiveConnectionChecker, MissingDataChecker, NewDirectiveChecker, OptimizedKPIChecker
from reporter import PredictedKPIChecker, NPSpecificConsChecker, NGSpecificConsChecker, seconds_to_ms

from utils import read_json, read_pickle, write_json

from validators import NOConnectionError, connection_check

from model_validators import define_reactor_hydrogen_diolefin_check, icc_htwo_aromatic_check, hydrogen_hc_check
from model_validators import OperationalKPIError,  OptimizationLowDifferenceError
from model_validators import calculate_percent_change,check_optimization_difference, track_different_tags
from similarity import search_dates

def parse_args() -> argparse.Namespace:
    """Function to parse command line arguments"""

    parser = argparse.ArgumentParser()

    parser.add_argument('input_dir', type=str, help='Path to the input folder', default='../Input-30min')
    parser.add_argument('output_dir', type=str, help='Path to the output folder', default='../Output')
    parser.add_argument('done_dir', type=str, help='Path to the done folder', default='../Done')
    
    # NOTE: the folders above need to be dynamic for the software team however the model and error folder can be hardcoded (static) 

    args = parser.parse_args()

    return args

def main():
    """Main Function"""
    # setup logger
    formatstr = '%(asctime)s: %(levelname)s: %(funcName)s Line: %(lineno)d %(message)s'
    datestr = '%m/%d/%Y %H:%M:%S'
    logging.basicConfig(
        level=logging.INFO, 
        format=formatstr, 
        datefmt=datestr, 
        handlers=[
            logging.FileHandler('data_download.log'),
            logging.StreamHandler()
            ]
        )

    # set up second logger to track missing variables
    handler = logging.FileHandler('missing.log')
    handler.setFormatter(logging.Formatter(fmt=formatstr, datefmt=datestr))

    missing_logger = logging.getLogger("missing_logger")
    missing_logger.setLevel(logging.WARNING)
    missing_logger.addHandler(handler)

    args = parse_args()

    logging.info("Reading config and loading model")

    # read configs any other files for setup
    config = read_json("../Model/info.json")
    reporter_info = read_json('../Model/reporter_info.json')
    final_model = read_pickle('../Model/xgb_pacol2_sasol_main_product_30mins.pkl')
    optimizer_info = read_json('../Model/p2_optimizer.json')
    previous_lims_ids = read_json('../Model/previous_lims.json')
    previous_lims_ids = {int(k):v for k,v in previous_lims_ids.items()}

    # NOTE: load additional configs for the similarity check
    short_to_long_map = read_json('../Model/short_to_long_name_map.json')

    # combine the all object ids into a single variable
    object_ids = [*config['controllable'], *config['noncontrollable'], *config['additional_ids'], *config['lims_ids']]

    # separator to use ...FIXME: figure out where to put this elsewhere
    sep = '___' 

    logging.info('Reading Mapper file')
    mh = MapperHandler('Datapoint Name', 'Datapoint Id', 'Property Name', 'Property Id')
    mapper, sub, name_to_id, id_to_name, final_object_ids, final_property_ids = mh.read_filter_create_extract(
        mapper_path=config['mapper_path'], object_ids=object_ids, property_ids=config['property_ids'], 
        *config['read_params']['args'], **config['read_params']['kwargs'])

    logging.info('Reading input files')
    reader = read_input_folder(args.input_dir, '.csv', object_ids=final_object_ids, property_ids=final_property_ids, id_to_name=id_to_name, sep=sep)

    # initialize reporters
    live_connection_checker = LiveConnectionChecker(*reporter_info['LiveConnection'])
    missing_tag_checker = MissingDataChecker(*reporter_info['MissingTagCount'])
    new_directive_checker = NewDirectiveChecker(*reporter_info['NewDirective'])
    predicted_kpi_checker = PredictedKPIChecker(*reporter_info['PredictedKPI'])
    optimized_kpi_checker = OptimizedKPIChecker(*reporter_info['OptimizedKPI'])
    np_specific_cons_checker = NPSpecificConsChecker(*reporter_info['NPSpecificCons'])
    ng_specific_cons_checker = NGSpecificConsChecker(*reporter_info['NGSpecificCons'])

    # Validation ID Sheet
    val_object_sheet = pd.read_csv(config['mapper_path_val'])
    # Filter Controllable 
    val_object_sheet = val_object_sheet.loc[:, ['Object Name', 'Object Id']][val_object_sheet['Object Name'].isin(list(optimizer_info['controllable'].keys()))].drop_duplicates()
    # Validation Output ID
    name_to_id_val = {(k, 'Value'): (v, 53)for k, v in zip(val_object_sheet['Object Name'], val_object_sheet['Object Id'])}

    # preallocate lists to hold incoming files and processed data ...
    files = []
    new_data = []
    no_connection_files = []

    for incoming_file, processed_data in reader:

        resample_data = processed_data.resample('60T', closed='right').mean()
        time = re.findall('\d{4,}_\d{2,}_\d{2,}_\d{2,}_\d{2,}', incoming_file)
        date_time = pd.to_datetime(time, format = "%Y_%m_%d_%H_%M")
        resample_data.index = date_time
        
        try:
            connection_check(resample_data)
        except NOConnectionError as e:
            print(e)
            # NOTE: add functionality to save file out if the data is missing and set the live connection boolean to false
            logging.info(f"Moving file: {incoming_file} to done folder and setting the live connection boolean to false")
            move(osp.join(args.input_dir, incoming_file), osp.join('../Error', incoming_file))
            # NOTE: if connection fails then update the checker
            epoch = resample_data.index[0].timestamp()
            live_connection_checker.add_value(epoch, 0)
            no_connection_files.append(resample_data)     
              
        else:
            files.append(incoming_file)
            new_data.append(resample_data)
            epoch = resample_data.index[0].timestamp()

    # read all files in and then concat them together in a single dataframe
    new_data = pd.concat(list(new_data))

    # load in previous data_vals
    previous_vals_df = pd.read_csv('../Model/previous_tag_vals.csv', parse_dates=['Date'], index_col='Date')
    
    # take mean of each tag
    mean_vals = pd.DataFrame(previous_vals_df.mean().to_dict(),index=[previous_vals_df.index.values[-1]])

    previous_lims = {'___'.join(id_to_name[obj_id, -6]):v for obj_id, v in previous_lims_ids.items()}

    # NOTE: remove lims from missing_count
    not_lims = [col for col in new_data.columns.tolist() if col not in list(previous_lims.keys())]

    for i in range(new_data.shape[0]):
        timestamp = new_data.index[i]
        missing_count = new_data.loc[timestamp, not_lims].isnull().sum()

        if missing_count > 0:
            missing_tag_checker.add_value(timestamp.timestamp(), missing_count)

        # add in checking and filling of lims data here
        for name in previous_lims.keys():
            # NOTE: do not add lims data to the missing variables count
            if np.isnan(new_data.at[timestamp, name]):
                new_data.loc[timestamp, name] = previous_lims[name]
            else:
                previous_lims[name] = new_data.at[timestamp, name]

        for col in new_data.columns:
            if np.isnan(new_data.at[timestamp, col]):
                missing_logger.warning(col.replace('___Value', ''))  # report missing variable to the logger
                new_data.loc[timestamp, col] = mean_vals.at[mean_vals.index[0], col].copy()

    
    missing_logger.warning('\n')  # log a blank line to space out different times
    all_vals = pd.concat([previous_vals_df, new_data])

    # save last 4 values
    last_vals = all_vals.iloc[-4:]
    last_vals.to_csv('../Model/previous_tag_vals.csv', index = True, index_label = 'Date')

    # update previous lims ... may or may not have been modified by the loop above
    previous_lims_ids = {str(name_to_id[tuple(full_name.split('___'))][0]):v for full_name, v in previous_lims.items()}
    write_json(previous_lims_ids, '../Model/previous_lims.json')

    logging.info('Generating Bounds')
    # Create Finalizied Bounds for optimization
    bounds_df = new_data[[var+'___Value' for var in list(optimizer_info['controllable'].keys())]]
    rate_change = [item['rate'] for item in optimizer_info['controllable'].values()]

    # Dynamic bounds
    min_init_bounds = bounds_df -  (bounds_df * rate_change)
    max_init_bounds = bounds_df +  (bounds_df * rate_change)

    # Global bounds
    min_bounds = [item['bounds'][0] for item in optimizer_info['controllable'].values()]
    max_bounds = [item['bounds'][1] for item in optimizer_info['controllable'].values()]


    for i, cont in enumerate(list(optimizer_info['controllable'].keys())):
        cont = cont + '___Value'
        min_init_bounds.loc[min_init_bounds[cont] < min_bounds[i], cont] = min_bounds[i]
        min_init_bounds.loc[min_init_bounds[cont] > max_bounds[i], cont] = max_bounds[i] - 0.0001
        max_init_bounds.loc[max_init_bounds[cont] < min_bounds[i], cont] = min_bounds[i] + 0.0001
        max_init_bounds.loc[max_init_bounds[cont] > max_bounds[i], cont] = max_bounds[i]

    # Collection of bounds    
    bounds = [
            [(min_init_bounds.at[timestamp, col+'___Value'], max_init_bounds.at[timestamp, col+'___Value']) for col in optimizer_info['controllable'].keys()] 
            for timestamp in new_data.index.tolist()
            ]
    
    # Minimum changing rate for controllables
    min_rate_change_threshold = [item['min_rate'] for item in optimizer_info['controllable'].values()]

    # run the high level algorithm here ... this can either be row by row 
    logging.info('Running model and optimization')
    # preallocate results 
    result = [None] * new_data.shape[0]

    # assign controllable variables and bounds --- this ensures the order is correct since the data is selected with this
    # TODO: handle the names of the live data being object_name{sep}property_name while the keys will just be object_name ...
    controllable_vars = list(optimizer_info['controllable'].keys())

    # NOTE: figure out how to handle this better --- appending the property here since this will be required to access the dataframe columns
    controllable_vars = [control + sep + "Value" for control in controllable_vars]
    noncontrollable_vars = [name + sep + "Value" for name in optimizer_info['noncontrollable']]
    # add failure row denoting no optimized values
    failure_row = [config['failure_label']] * len(controllable_vars)

    time_dif_tags_dict = {}

    for i in range(new_data.shape[0]):
        timestamp = new_data.index[i]
        # copy variables to make sure there is no reference issue
        controls = new_data.loc[timestamp, controllable_vars].copy().values
        noncontrollable = new_data.loc[timestamp, noncontrollable_vars].copy().values

        # calculate NP Specific Cons 
        np_specific_value = 100 * new_data.at[timestamp, 'T8.PACOL2.6FIC-403 Augusta___Value'] / (new_data.at[timestamp, 'T8.PACOL2.6FI-474 Augusta___Value']*new_data.at[timestamp, 'V406-TOTALI N-OLEFINE Augusta___Value'])
        np_specific_cons_checker.add_value(timestamp.timestamp(), np_specific_value)

        # calculate NG Specific Cons 
        ng_specific_value = new_data.at[timestamp, 'T8.PACOL2.6FI-120_51 Augusta___Value'] / new_data.at[timestamp, 'T8.PACOL2.6FI-440 Augusta___Value']
        ng_specific_cons_checker.add_value(timestamp.timestamp(), ng_specific_value)

        # Import Dynamic Bounds
        d_bounds = bounds[i]

        # Run Optimization Here
        optimal_controls, success = run_model(
            controllable=controls, noncontrollable=noncontrollable, bounds=d_bounds, 
            final_model=final_model, **optimizer_info['kwargs'])

        original_values = np.array(np.concatenate([controls, noncontrollable])).reshape(1,-1)
        optimal_values = np.array(np.concatenate([optimal_controls, noncontrollable])).reshape(1,-1)

        original_product = final_model.predict(original_values)[0]
        optimized_product = final_model.predict(optimal_values)[0]

        predicted_kpi_checker.add_value(timestamp.timestamp(), original_product)        
        optimized_kpi_checker.add_value(timestamp.timestamp(), optimized_product)

        # calculate change between optimal and orinal values
        percent_change = calculate_percent_change(optimal_controls, controls)

        # figure out which tags are different
        different_tags = track_different_tags(percent_change, controllable_vars, min_rate_change_threshold)

        # Domain Fix Only give directives for T8.PACOL2.6FIC-1402_A Augusta___Value when it's greater than 3
        if new_data.at[timestamp, 'T8.PACOL2.6FIC-1402_A Augusta___Value']<3:
            try:
                different_tags.remove('T8.PACOL2.6FIC-1402_A Augusta___Value')
            except:
                pass

        if new_data.at[timestamp, 'T8.PACOL2.6FIC-1402 Augusta___Value']<3:
            try:
                different_tags.remove('T8.PACOL2.6FIC-1402 Augusta___Value')
            except:
                pass

        # Domain Fix Only increase T8.PACOL2.6TIC-403 Augusta___Value when abs difference between (6TI-400_4-6TI-400_6) is decreasing
        # Notice the 6TIC-403 must be the last controllable tag in the model
        if (np.abs(new_data.at[timestamp, 'T8.PACOL2.6TI-400_4 Augusta___Value'] - new_data.at[timestamp, 'T8.PACOL2.6TI-400_6 Augusta___Value'])>0) & (optimal_controls[-1]>controls[-1]):
            try:
                different_tags.remove('T8.PACOL2.6TIC-403 Augusta___Value')
            except:
                pass
        
        # Domain Fix Only increase 6FIC-478 when T8.PACOL2.6PIC-412 Augusta___Value is less than or equal to 0.45
        if (new_data.at[timestamp, 'T8.PACOL2.6PIC-412.PID_RESULT Augusta___Value'] > 42) & (optimal_controls[2]>controls[2]):
            try:
                different_tags.remove('T8.PACOL2.6FIC-478 Augusta___Value')
            except:
                pass
        
        time_dif_tags_dict.update({timestamp: []})

        # try:
        #     check_final_product(original_product, optimized_product)

        #     # Comment this line 02/02/2023    seems to be duplicated
        #     # check_optimization_difference(percent_change, controllable_vars)

        # except DecreasingProduct as e:
        #     logging.warning(f'Timestamp {timestamp} failed:\t' + str(e))
        #     move(osp.join(args.input_dir, files[i]), osp.join("../Error", files[i]))
        #     result[i] = failure_row
        #     continue  # NOTE: confirm this makes sure nothing in this else block executes after this ...

        
        optimal_control_df = pd.DataFrame(dict(zip(controllable_vars, optimal_controls)), index = [timestamp])

        # NOTE: hack so scripts aren't dependent on names but ids instead
        lims_tags = list(previous_lims.keys())
        additional_tags = ['T8.PACOL2.6FI-1401_B Augusta___Value', lims_tags[0]]
        
        define_kpi_tags = ['T8.PACOL2.6FIC-1402 Augusta___Value', 'T8.PACOL2.6FIC-1402_A Augusta___Value', 'T8.PACOL2.6FI-1401_B Augusta___Value', 
                           lims_tags[0]]
        # INGRESSO R490-DIOLEFINE Augusta = lims_tags[0]

        # TODO: FIX 3rd element
        icc_hyd_arom_tags = ['T8.PACOL2.6FIC-478 Augusta___Value', 'T8.PACOL2.6FIC-479_1 Augusta___Value', 'INGRESSO R473-AROMATICI UOP Augusta___Value']
        # TODO: Fix 1st element
        reactor_hyd_hc_tags = ['K401-HYDROGEN Augusta___Value', 'T8.PACOL2.6FI-401 Augusta___Value', 'T8.PACOL2.6TI-400_11 Augusta___Value', 'T8.PACOL2.6PI-406 Augusta___Value', 'T8.PACOL2.6TI-400_11 Augusta___Value', 'T8.PACOL2.6PI-406 Augusta___Value', 'T8.PACOL2.6FIC-402 Augusta___Value']

        optimal_control_merge_df = optimal_control_df.merge(new_data.loc[[timestamp], additional_tags], how='inner', right_index=True, left_index=True)
        define_kpi_values = optimal_control_merge_df.loc[timestamp, define_kpi_tags].copy()

        icc_hyd_arom_df = optimal_control_df.merge(new_data.loc[[timestamp], [tag for tag in icc_hyd_arom_tags if tag not in controllable_vars]], how = 'inner', right_index = True, left_index = True)
        icc_hyd_arom_values = icc_hyd_arom_df.loc[timestamp, icc_hyd_arom_tags].copy()

        reactor_hyd_hc_df = optimal_control_df.merge(new_data.loc[[timestamp], reactor_hyd_hc_tags], how = 'inner', right_index = True, left_index = True)
        reactor_hyd_hc_values = reactor_hyd_hc_df.loc[timestamp, reactor_hyd_hc_tags].copy()



        # DeFine Reactor Hydrogen to Diolefin Molar Ratio
        # .3 might be added placeholder for mass percent of hydrogen in these streams ... this is not really constant and it is related to the catalyst lifecycle 

        # (T8.PACOL2.6FIC-1402 Augusta + T8.PACOL2.6FIC-1402_A Augusta) * .3 / .00201
        molar_hydrogen = (define_kpi_values[0] + define_kpi_values[1]) * 0.3 / 0.00201  # addition of two hydrogen streams

        # 10 * T8.PACOL2.6FI-1401_B Augusta * INGRESSO R490-DIOLEFINE Augusta / .15813 ... 
        # assuming lims is a percent (decimal) and molar mass is in kg/mol
        molar_diolefin = 10 * define_kpi_values[2] * define_kpi_values[3] / 0.15813  # constant is to convert to moles
        define_kpi = molar_hydrogen / molar_diolefin

        # ICC H2/Aromatic KPI
        hydrogen_molar_flow_rate = (icc_hyd_arom_values[0]/0.0201)
        aromatic_molar_flow_rate = (1000 * icc_hyd_arom_values[1] * (icc_hyd_arom_values[2]/100))/0.156
        icc_hydrogen_aromatic_kpi = hydrogen_molar_flow_rate/aromatic_molar_flow_rate

        # Reactor Hydrogen/HC KPI
        ph_two = -0.00060193 * reactor_hyd_hc_values[4] + 0.0652977 *  reactor_hyd_hc_values[5] + 0.12507
        h_two_molar_flow_rate = (reactor_hyd_hc_values[0]/100) * (reactor_hyd_hc_values[1] * (101325/273.15) * ((reactor_hyd_hc_values[2] + 273.15)/( (reactor_hyd_hc_values[3] + 1) * 10**5))) * (ph_two/0.00201)
        hc_molar_flow_rate = (reactor_hyd_hc_values[6] * 100)/0.162134
        reactor_hydrogen_hc_kpi = h_two_molar_flow_rate/hc_molar_flow_rate

        # KPI error check
        # KPI works in local, but does not work on server  01/12/2023

        try:
            pass
            # define_reactor_hydrogen_diolefin_check(define_kpi)
            # icc_htwo_aromatic_check(icc_hydrogen_aromatic_kpi)
            # hydrogen_hc_check(reactor_hydrogen_hc_kpi)
        except (OperationalKPIError) as e:
            logging.warning(f'Timestamp {timestamp} failed:\t' + str(e))
            move(osp.join(args.input_dir, files[i]), osp.join("../Error", files[i]))
            result[i] = failure_row
            
        else:
            if success:
                result[i] = optimal_controls
                move(osp.join(args.input_dir, files[i]), osp.join(args.done_dir, files[i]))
                new_directive_checker.add_value(timestamp.timestamp(), 1)
                time_dif_tags_dict.update({timestamp: different_tags})

            else:
                logging.warning(f"The timestamp {timestamp} did not converge")
                result[i] = failure_row
                move(osp.join(args.input_dir, files[i]), osp.join("../Error", files[i]))

    # now final results need to be written out:
    # the final variable should be a dataframe with the timestamp as the index and the objectName{sep}PropertyName as the columns
    # the contents of the files saved will only be the directives --- not any kpi values or byproduct values ...
    final: pd.DataFrame = pd.DataFrame(result, columns=controllable_vars, index=new_data.index) # this should be the end result of running the optimization

    historical_data = pd.read_csv('../Model/final_historical_data.csv', parse_dates=['Date']).set_index('Date')
    historical_data = historical_data.rename(columns=short_to_long_map)
    historical_data.columns = [col + '___Value' for col in historical_data.columns.tolist()]


    similar_datapoints = {}
    for i, timestamp in enumerate(final.index):
        # TODO: swap out the controllable_vars with the more general tags_of_interest
        this_match  = search_dates(historical_data.loc[:, controllable_vars], final.loc[timestamp, controllable_vars])
        similar_datapoints[timestamp] = this_match

    info_reporter = AdditionalInfoReporter([live_connection_checker, missing_tag_checker, new_directive_checker, np_specific_cons_checker, ng_specific_cons_checker, predicted_kpi_checker, optimized_kpi_checker])
    out_prefix = 'data_pacol2_'
    logging.info('Saving results')
    for i, timestamp in enumerate(final.index):

        ## Note: Error lies in here!!!
        live_format, columns, file_name, this_epoch = put_data(final.loc[[timestamp], time_dif_tags_dict[timestamp]], args.output_dir, name_to_id, sep=sep, output_property_id=config['output_property_id'])
        # get all the additional info 
        additional_info = info_reporter.report(epoch=this_epoch)
        # append it to the ds info
        live_format.extend(additional_info)

        # Find validation point 
        matches = similar_datapoints[timestamp]
        if matches.shape[0] != 0 and new_directive_checker.value[timestamp.timestamp()] != 0:
            matches.rename(index={matches.index[0]: timestamp},inplace=True)
            val_live_format, _, _, _ = put_data(matches, args.output_dir, name_to_id_val, sep=sep, output_property_id=53)
            live_format.extend(val_live_format)
        
        # format to df and save
        out = pd.DataFrame(live_format, columns=columns)
        file_path = osp.join(args.output_dir, out_prefix+file_name)
        out.to_csv(file_path, sep=';', header=False, index=False)

    for i, this_file in enumerate(no_connection_files):
        live_format, columns, file_name, epoch = put_data(this_file.iloc[[0], :].loc[:, controllable_vars], args.output_dir, name_to_id, sep=sep, output_property_id=config['output_property_id'])
        # get all the additional info for this timestamp
        additional_info = info_reporter.report(epoch=epoch)
        # append it to the ds info
        live_format.extend(additional_info)
        # format to df and save
        out = pd.DataFrame(live_format, columns=columns)
        file_path = osp.join(args.output_dir, out_prefix+file_name)
        out.to_csv(file_path, sep=';', header=False, index=False)


    return

if __name__ == '__main__':
    main()
