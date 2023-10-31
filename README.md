# SumoPred-PLM: human SUMOylation and SUMO2/3 sites Prediction using Pre-trained Protein Language Model

To guarantee proper use of our tool, please follow all the steps in the presented order.
<br>
<br>
>[!Note]
> All the programs in this repository are properly tested and accurate for its functionality! 
## How it Works
SumoPred-PLM is a multi-layer perceptron-based approach that leverages contextualized embeddings generated
from the ProT5-XL-UniRef50 Protein Language Model (Referred to here as `ProT5 PLM`) to predict Human SUMOylation and SUMO2/3 sites (`"K"`).

## Environment Details

SumoPred-PLM was developed in the following environment:

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

Please ensure you have a compatible Python environment with these dependencies to use the model effectively.

## Create a Virtual Environment

To install the required dependencies for this project, you can create a virtual environment and use the provided `requirements.txt` file. Make sure your Python environment matches the specified versions above to ensure compatibility.

```
python -m venv myenv  # Create a virtual environment
source myenv/bin/activate  # Activate the virtual environment
pip install -r requirements.txt  # Install the project dependencies
```

## Download Protein Sequences From [Uniprot.org](https://www.uniprot.org/)  

You can use our model with all human protein sequences provided by UniPort.
  - We have included a file named `P10275.fasta` as an example in this repository

Once you have downloaded the protein sequence you can send the sequence as input to the ProT5 PLM.


## Generate Contextualized Embeddings Using ProT5 PLM

### ProT5 PLM Installation 

In order to run the ProT5 PLM, you need to install the following modules:
```bash
pip install torch
pip install transformers
pip install sentencepiece
```

For more information, refer to this ["ProtTrans"](https://github.com/agemagician/ProtTrans) Repository.


Once the model is installed you can write your own code to generate the contextualized embeddings **OR**
to make this process easier for you, we have also provided a `Generate_ProtT5_Contextualized_Embeddings_of_P10275_Protein.ipynb` file.

Here's how you can use the provided file for a single protein sequence:

You need to make changes to only two lines of the code:
```
basedir = "/Generate_ProtT5_File/after_cd_hit_files"       # here you will paste the location where your .fasta file is located
name = "P10275.fasta"                                      # here you will add your protein name
```
The rest of the lines will stay the same.

Once the code is successfully run, It will generate a file named `"P10275_Prot_Trans_.csv"`.

Here's how the output file will look:

<img max-width = 100% alt="image" src="https://github.com/PakhrinLab/OglyPred-PLM/blob/main/images/ProT5_Output.png">
<br>

Rows = Length of The Protein Sequence  
Columns = 1025 (1 column to represent the protein residue + 1024 embeddings)
<br>

>[!NOTE]
>Make sure to keep all of the files being used in the same directory


## Extracting "K" Sites From Contextualized Embeddings

In order to extract only the "K" sites from the contextualized embeddings you can use the following code:

``` bash
import pandas as pd
df = pd.read_csv("P10275_Prot_Trans_.csv", header = None)                     # replace with your ProtT5 embeddings file
Header = ["Residue"]+[int(i) for i in range(1,1025)]
df.columns = Header
df_K_Residue_Only = df[df["Residue"].isin(["K"])]
df_K_Residue_Only.to_csv("P10275_K_Sites.csv", index = False)                 # saves the embeddings of only K residues

```

Once the process is complete, you will have a .csv file containing the embeddings of "K" sites.

Here's an example output:

<img max-width = 100% alt="image" src="https://github.com/PakhrinLab/OglyPred-PLM/blob/main/images/Extraction_S_T_Ouput.png">
<br>

## Sending Sites into SumoPred-PLM For Prediction

Now send the `P10275_K_Sites.csv` from the previous step as an input to the SumoPred-PLM.

We have provided a file named `SumoPred-PLM.ipynb` and
our model`Subash_Chandra_Pakhrin5775457372.h5` which should be downloaded and kept in the same directory to avoid any issues.


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
