## Sasol Pacol2 Yield Optimization

#### Pacol2 Workflow

```mermaid

graph LR


I.C.C --> Pacol
Pacol --> Define
Define -- 6FI-474 * USCITA R7003 Olefin Concentration --> Stripper
Injected_Olefin -- 8FIC-7016 --> Stripper
Stripper -- Olefin*K_HF --> HF
HF -- 8FIC-7019 --> I.C.C
Stripper -- Olefin*K_DT --> Detal
Detal -- 8FI-7016 --> I.C.C
	
```

$$ Target: Max(Olefin \ Yield) $$

> Olefin Yield = 8FI-7024_B * USCITA R7003 Olefin Concentration

* Build relationship between:

	$$ Olefin = f(C_1, C_2, ..., C_n, NC_1, NC_2, ..., NC_n)$$
	* C means controllable variables
	* NC means non-controllable variables

* Optimization:
	
	* Use dual_annealing to do the optimization by tweaking the value of C.
	* By tweaking, we need a range for each C.
	* Find Global upper & lower bounds for each C from historical data.
	* If the upper & lower bounds for C vary too much, an investigation must be done. 
	*  Apply max & min changing rates for each C.
	* Min changing rates should be there because the client does not want to do minor change. 
	* Max changing rates should be there because too drastic changes cannot be achieved. 

## Data Science Workflow

#### Historical Analysis

```mermaid

graph LR

Maestro -- Tag list -- Sql Module --> DataSet{DataSet}

subgraph EDA.ipynb 

DataSet --> EDA
EDA --> FeatureEngineering(Feature Engineering)

end

subgraph Feature Selection & Baseline.ipynb

FeatureEngineering --> FeatureSelection(BorutaShap Feature Selection)
FeatureSelection <-- refined tag list --> IES((IES))
IES <-- verified tag list --> FeatureSelection
FeatureSelection -- verified tag list --Validation Strategy --> BaselineModel{BaselineModel}

end 

BaselineModel -- MLFlow -- Tuning -- Validation --> FinalModel{Final Model}

subgraph Config_gen.ipynd

FinalModel --> Config{Config}

end

```

#### Config explanation (From historical to Live)

```mermaid
classDiagram

Optimizer <--Final Model : config_gen.ipynb
Optimizer: controllable dict
Optimizer: noncontrollable list
Optimizer: additional list
Optimizer: lims list
Optimizer: kwargs()

Optimizer --> controllable : Elaborate
controllable: key Tag_name
controllable: value Tag_detail

controllable --> Tag_detail : Elaborate
Tag_detail: max_changing_rate float
Tag_detail: min_changing_rate float
Tag_detail: global_bounds list

Optimizer --> UsedDatapointNames : Generate
UsedDatapointNames: Full_Tag_Name csv

Optimizer --> Info : config_gen.ipynb
Info: controllable_id list
Info: noncontrollable_id list
Info: additional_id list
Info: Lims_id list
Info: mapper path str
Info: output_property_id -180
Info: property_id -6 
Info: read_params(kwargs**)

Info --> Live_Data: data_mapper package

```

```mermaid
class  Optimizer.json{controllable:{
									 Tag_name_A:{
									             'max_changing_rate': float,
									             'min_chaning_rate': float,
									             'global_bounds': list[float, float]
									              },... 
									 },
					   noncontrollable: list[tag_name, tag_name,...],
					   additional: list[tag_name, tag_name,...],
					   lims: list[tag_name, tag_name,...],
					   kwargs:{}
					   }

** additional is the additional tags involved in the process(KPI related, other miscellaneous).
** Lims is the Lims tags involved in the process(Concentrations).

class Info.json{
                read_params: {
			                  args:,
			                  kwargs:{sheet_name: str
			                          },
			                  }
			    mapper_path: str,
			    mapper_path_val: str,
			    output_property_id: -180,
			    property_ids: [-6],
			    controllable_id: [int, int,...],
			    noncontrollable_id: [int, int, ...],
			    additional_id: [int, int, ...],
			    Lims_id: [int, int, ...],
                }

class UsedDatapointNames.csv: all the used tags' name in the process
```
#### Live Data Explanation

Incoming live data (raw data)

|Object ID |Property ID |Value	|Timestamp	|0                  
|----------------|--------------------|---------------|------------|------------
|40293	|-6	|1.85	|1675879050000|0
|40336	|-6	|3720	|1675879050000|0
|...	|...|...	|...|0

Output live data (directives)
|Object ID |Property ID |Value	|TimeStamp	|0                  
|----------------|--------------------|---------------|------------|------------
|40424	|-180	|21.31	|1675879050000|0
|40475	|-180	|133.21	|1675879050000|0
|31300	|27	|1.012	|1675879050000|0
|...	|...|...	|...|0

> All the object/property id can be found in a spread sheet given by R&D. Should put the data into the right place. 
> When pulling\ submitting data from\to __Maestro__, these data should be in __live format__.

#### Maestro demo

1. Look at the trend of a specific tag.
2. Final Dashboard. 

#### Live Deployment

```mermaid

graph LR

subgraph main.py

LiveData{Live Data} -- data mapper package --> DataFrame{DataFrame}
TagInformation{Tag Information} --> DataFrame{DataFrame}
DataFrame{DataFrame} -- imputation--> DataFrame{DataFrame}
local/global_bounds -- bounds --> Optimization
DataFrame{DataFrame} -- before optimization --> Optimization
Optimization --> ActualOutput{Actual Ouput}
Optimization --> OptimizedOutput{Optimized Ouput}
Optimization --> Directives{Directives}
Directives{Directives} --> KPI[KPI/Domain filter]
KPI[KPI/Domain filter] -- No --> Drop(drop)
KPI[KPI/Domain filter] -- Yes --> Validation[Validation Point]
ActualOutput{Actual Ouput} -- checker --> LiveOutput{Live Output}
OptimizedOutput{Optimized Ouput} -- checker --> LiveOutput{Live Output}
Validation[Validation Point] -- data mapper --> LiveOutput{Live Output}
Directives{Directives} -- data mapper --> LiveOutput{Live Directives Output}

end
```

> For more details, look at repo [HF-Live](https://github.com/Elutions/hf-live) with good explanation. 



> Written with [StackEdit](https://stackedit.io/).
