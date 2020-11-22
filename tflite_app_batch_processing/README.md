# tflite/App Batch Processing

### Table of Contents
1. Quick Start
2. Related Notes

By allowing classification of images via a file-of-images-files, batch processing allows for higher throughput classification than by the UI and photos.  Batch processing is initated via the app itself.  Batch processing is implemented in the "Batch" screen of the app and by web servers.  The execution of Batch processing is in the *Quick Start* section below.

# Quick Start

**First**, in this directory run the python3 web server command:
```
python3 -m http.server
```
This will start a basic web server on port 8000 so that the file img_list_louis_test.txt can be served via the URL http://192.168.1.76:8000/img_list_louis_test.txt where 192.168.1.76 is the IP of the server.  This is the default value for the FOF URL in the batch screen of the app.  While the APP is running the user may need to touch the URL to modify it to conform to the IP of the server.  This server will also serve the images.  __Note__ that the images (or links to them) will need to be copied to this directory so that the files are accessible.  Otherwise, *404* errors will result.  The paths to the images must conform to the paths as specified in the file img_list_louis_test.txt.  Further, **note** that a file's path indicates its 'true' label.  In the txt file for example the image indicated by the path "Louis_Data/Test/Cheetah/M Alvin/Alvin-2.jpg" is known to be a Cheetah because "Cheetah" is in the path leading to the file and matches an entry in the file "species_13_labels.txt".

**Second**, also in this directory run the command:
```
FLASK_APP=image_reporter.py flask run --host=0.0.0.0
```
This will start a Flask server to handle HTTP GET requests whose URLs contain the results of the classifications from the image files in the FOF loaded from the regular http server above.  As batch processing and image classification proceeds, the flask server receives the requests and *appends* to the file report_db.tsv writing results there.  Because it appends, if multiple batch runs are necessary manual editing of the report_db.tsv file may be needed so as to eliminate duplicate headers in the middle of the file.

**Third** after the above servers are started, run the app, go to the "Batch" screen and make sure the URLs are set to use the above servers, and then hit the "Classify Batch" button.

**Fourth**, after a successful batch processing use the Jupyter notebook **Explore_results.ipynb** to read the report_db.tsv file as well as the species_13_labels.txt to generate prediction results figures and tables (e.g. confusion matrix).

# Related Notes
python3 is required to run these scripts.  Jupyter is also required to run the notebook.  In addition the following non-standard but well-known modules are required:

* Flask
* sklearn
* matplotlib
* pandas
* seaborn

The file species_13_labels.txt has entries with *whitespace*.  At present this whitespace is known and code accounts for it.  To the best of my knowledge, as I write this, the code uses this whitespace and if the file is modified to remove the whitespaces, then problems may arise.  In addition, the paths in img_list_louis_test.txt also have *whitespace* and if the whitespace is removed from there then problems may also ensue.  For this reason, it is recommended to conform to any whitespace in the labels and in the paths.  Otherwise, some parsing code needs to be re-written.

