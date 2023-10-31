# SumoPred-PLM: human SUMOylation and SUMO2/3 sites Prediction using Pre-trained Protein Language Model.

# OglyPred-PLM 
Human *O*-linked Glycosylation Site Prediction Using Pretrained Protein Language Model

To guarantee proper use of our tool, please follow all the steps in the presented order.
<br>
<br>
>[!Note]
> All the programs provided in this repository are properly tested and accurate for its functionality! 
## How it Works
OglyPred-PLM is a multi-layer perceptron based approach that leverages contexualized embeddings generated
from the ProT5-XL-UniRef50 Protein Language Model (Referred here as `ProT5 PLM`) to predict Human O-linked 
glycosylation sites (`"S/T"`).

## Environment Details

OglyPred-PLM was developed in the following environment:

- Python version: 3.8.3.final.0
- Python bits: 64
- Operating System: Linux
  - OS release: 5.8.0-38-generic
- Machine architecture: x86_64
- Processor architecture: x86_64

### Python Packages and Dependencies

The model relies on the following Python packages and their specific versions:

- Pandas: 1.0.5
- NumPy: 1.18.5
- Pip: 20.1.1
- SciPy: 1.4.1
- scikit-learn: 0.23.1
- TensorFlow: 2.3.1

Please ensure that you have a compatible Python environment with these dependencies to use the model effectively.

## Create a Virtual Enviornment

To install the required dependencies for this project, you can create a virtual environment and use the provided `requirements.txt` file. Make sure your Python environment matches the specified versions above to ensure compatibility.

```
python -m venv myenv  # Create a virtual environment
source myenv/bin/activate  # Activate the virtual environment
pip install -r requirements.txt  # Install the project dependencies
```

## Download Protein Sequences From [Uniprot.org](https://www.uniprot.org/)  

You can use our model with all human protein sequences provided by UniPort.
  - We have included a file named `Q63HQ2.fasta` as an example in this repository

Once you have downloaded the protein sequence you can send the sequence as input to the ProT5 PLM.


## Generate Contexualized Embeddings Using ProT5 PLM

### ProT5 PLM Installation 

In order to run the ProT5 PLM, you need to install the following modules:
```bash
pip install torch
pip install transformers
pip install sentencepiece
```

For more information, refer to this ["ProtTrans"](https://github.com/agemagician/ProtTrans) Repository.


Once the model is installed you can write your own code to generate the contexualized embeddings **OR**
to make this process easier for you, we have also provided a `Generate_Contexualized_Embeddings_ProT5.ipynb` file.

Here's how you can use the provided file for a single protein sequence:

You need to make changes to only two lines of the code:
```
basedir = "/project/pakhrin/salman/after_cd_hit_files"     # here you will paste the location where your .fasta file is located
name = "Q63HQ2.fasta"                                      # here you will add your protein name
```
Rest of the lines will stay the same.

Once the code is successfully run, It will generate a file named `"Q63HQ2_Prot_Trans_.csv"`.

Here's how the output file will look:

<img max-width = 100% alt="image" src="https://github.com/PakhrinLab/OglyPred-PLM/blob/main/images/ProT5_Output.png">
<br>

Rows = Length of The Protein Sequence  
Columns = 1025 (1 column to represent the protein residue + 1024 embeddings)
<br>

>[!NOTE]
>Make sure to keep all of the files being used in the same directory


## Extracting "S/T" Sites From Contexualized Embeddings

In order to extract only the "S/T" sites from the contexualized embeddings you can use the following code:

``` bash
import pandas as pd

df = pd.read_csv("Q63HQ2_Prot_Trans_.csv", header = None)                     # replace with your ProtT5 embeddings file
Header = ["Residue"]+[int(i) for i in range(1,1025)]
df.columns = Header
df_S_T_Residue_Only = df[df["Residue"].isin(["S","T"])]
df_S_T_Residue_Only.to_csv("Q63HQ2_S_T_Sites.csv", index = False)            # saves the embeddings of only S and T residues

```

Once the process is complete, you will have a .csv file containing the embeddings of "S/T" sites.

Here's an example output:

<img max-width = 100% alt="image" src="https://github.com/PakhrinLab/OglyPred-PLM/blob/main/images/Extraction_S_T_Ouput.png">
<br>

## Sending Sites Into OglyPred-PLM For Predection

Now send the `Q63HQ2_S_T_Sites.csv` from the previous step as an input to the OglyPred-PLM.

We have provided a file named `OglyPred-PLM.ipynb` and
our model`Prot_T5_my_model_O_linked_Glycosylation370381Prot_T5_Subash_Salman_Neha.h5`which should be downloaded and kept in the same directory to avoid any issues.


## Any Questions?

If you need any help don't hesitate to get in touch with Dr. Subash Chandra Pakhrin (pakhrins@uhd.edu)

























Programs were executed using anaconda version: 2020.07, recommended to install the same

The programs were developed in the following environment. python : 3.8.3.final.0, python-bits : 64, OS : Linux, OS-release : 5.8.0-38-generic, machine : x86_64, processor : x86_64, pandas : 1.0.5, numpy : 1.18.5, pip : 20.1.1, scipy : 1.4.1, scikit-learn : 0.23.1., keras : 2.4.3, tensorflow : 2.3.1.

Please place all the following files in the same directory
  SUMOylation Independent Testing.ipynb,
  Final_independent_test_dataset_of_SUMOylation_PTM.csv (In the publicly shared google drive),
  Subash_Chandra_Pakhrin5775457372.h5,
  and execute the SUMOylation Independent Testing.ipynb program to see the reported result.

Please place all the following files in the same directory
  SUMO_2_3_Independent_Test_Set_Testing.ipynb,
  Independent_Test_Set_of_Hendriks_et_al_SUMO_2_3.csv (In the publicly shared google drive),
  SUMO2_330482327.h5,
  and execute the SUMO_2_3_Independent_Test_Set_Testing.ipynb program to see the reported result.
  
Please place all the following files in the same directory
  GPS SUMO Independent Testing.ipynb,
  GPS_Testing_Prot_T5_feature.txt (In the publicly shared google drive),
  GPS_SUMO_Independent_test82423____13.h5,
  and execute the GPS SUMO Independent Testing.ipynb program to see the reported result.
 
If you want to test if the sites of a protein is SUMOylated or not:
   Please place all the following files in the same directory
   SUMOylation_P10275_Test.ipynb,
   Subash_Chandra_Pakhrin5775457372.h5,
   P10275_Prot_Trans_.csv (Change this feature file with your interest, generate it using the analyze_Cell_Mem_ER_Extrac_Protein.py program)
   and execute the SUMOylation_P10275_Test.ipynb you will see the result which K are SUMOylated.
   
If you want to test if the sites of a protein is SUMO2/3 SUMOylated or not:
   Please place all the following files in the same directory
   SUMO_2_3_P10275_Test.ipynb,
   SUMO2_330482327.h5,
   P10275_Prot_Trans_.csv (Change this feature file with your interest, generate it using the analyze_Cell_Mem_ER_Extrac_Protein.py program)
   and execute the SUMO_2_3_P10275_Test.ipynb you will see the result which K are SUMO2/3 SUMOylated.
   
  *** For your convenience we have uploaded the ProtT5 feature extraction program (analyze_Cell_Mem_ER_Extrac_Protein.py) for the protein sequence from ProtT5 as well as corresponding 1024 feature vector extraction program (Feature Extraction Program from the generated files.ipynb) from the ProtT5 file. ***

All the training and independent test data are uploaded at the following google drive link: https://drive.google.com/drive/folders/1GsTRSQc2vwWH-tzBbkrXwrec6LpWUKhC

If you need any help please contact Dr. Subash Chandra Pakhrin (pakhrins@uhd.edu) 
