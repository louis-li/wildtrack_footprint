#!/usr/bin/python3

# a simple Flask app to record predictions from the
# android device/emulator
from flask import Flask
from flask import request
import sys
app = Flask(__name__)

report_local_file='report_db.tsv'

if __name__ == '__main__':
	print("Usage : FLASK_APP=image_reporter.py flask run --host=0.0.0.0")
	print("Upload results via HTTP GET query parameters in the URL http://localhost/report_image?run_id=someting&file_id=someting&model_name=someting&result=someting&start_time=someting&end_time=someting&file=something")
	print("Don't forget to URL encode the query parameters.")
	print("Results saved at "+report_local_file)
	sys.exit(0)
	
#add Headers to the report file by appending
report_headers=['run_id','file_id','model_name','result','start_time','end_time','file']
writer=open(report_local_file,'a')
writer.write('\t'.join(report_headers)+"\n")
writer.close()


@app.route('/report_image',methods=['GET'])
def record_prediction():
	"""
	Receive prediction results append them to the database/report file
	"""
	row=[request.args.get(r) for r in report_headers]
	to_write="\t".join(row)+"\n"
	writer=open(report_local_file,'a')
	writer.write(to_write)
	writer.close()
	return to_write


