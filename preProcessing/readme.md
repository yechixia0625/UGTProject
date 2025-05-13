How to preprocess the dataset?  
1. Downloading the dataset : https://c3.ndc.nasa.gov/dashlink/resources/1018/ save it in this directory.  
2. Design the 6-client distribution of the dataset and complete 3 ".xlsx" files in the "/template" directory.  
3. Change the relative file path in the "datasetsExtract.py" and run it.  
4. Change the relative file path in the "dataSort.py" and run it.  
5. You now have a folder with data for 6 clients, and that's the dataset!  
  
How to normalise the dataset?  
1. Run max_min.py to get the maximum and minimum values for each class for each client.  
2. Run.. / csv2png.m in Matlab for dataset normalization  
3. Run clearCSV.py to delete the files which contains maximum and minimum values in each client.  
4. Enjoy your dataset :)  