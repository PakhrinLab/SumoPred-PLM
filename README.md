# pLMSumoPred: Prediction of human SUMOylation and SUMO2/3 sites using embeddings from a pre-trained protein language model

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
