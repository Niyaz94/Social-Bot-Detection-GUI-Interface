from flask import Flask, jsonify, request,Response, flash, redirect, url_for,abort, render_template,send_file,make_response
import pandas as pd
import sklearn
import json

from project.model.PreProcessing import PreProcessing
from project.model.PredictModel  import PredictModel


ALLOWED_EXTENSIONS = {'json', 'csv'}

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and  filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['POST'])
def index():
    try:
        #Reading other parameters
        returnType, inputFileType, outputFileType, mlModel, featureNumber, applyOUSampling, datasetVersion, outputFormat, attachInputToOutput= map(request.form.get, ("type","inputFileType","outputFileType","mlModel","featureNumber","applyOUSampling","datasetVersion","outputFormat","attachInputToOutput"))

        #Uploading File
        filesList=request.files
        dfFile=filesList['file']
        if 'file' not in filesList or dfFile.filename == '' or not allowed_file(dfFile.filename):
            return redirect(404)

        applyOUSamplingLabel="un" if applyOUSampling==True else "fu"

        fileExtention=dfFile.filename.rsplit('.', 1)[1].lower()

        if inputFileType=="csv" and fileExtention=="csv":
            orignal_X=pd.read_csv(dfFile,lineterminator='\n')
        elif inputFileType=="json" and fileExtention=="json":
            orignal_X=pd.read_json(dfFile)
        else:
            return redirect(404)



        X=orignal_X.copy()
        preprocess=PreProcessing(X)
        check_preprocess=preprocess.check_output()

        if len(check_preprocess[0])>0 or len(check_preprocess[1])>0:
            return redirect(404)
          
        
        X=preprocess.output()

        predict=PredictModel(mlModel,datasetVersion,featureNumber,applyOUSamplingLabel,X)
        


        if returnType=="show":
            return json.dumps({
                "data":predict.TSNE(),
                "stat":predict.output()
            })   
        elif returnType=="download":

            if attachInputToOutput.lower()=="true":
                result= pd.concat([pd.DataFrame(predict.class_output(),columns=["class"]),orignal_X],axis=1)
            else:
                result=pd.DataFrame(predict.class_output(),columns=["class"])

            if outputFileType=="csv":
                resp = make_response(result.to_csv(index=False))
                resp.headers["Content-Disposition"] = "attachment; filename=download.csv"
                resp.headers["Content-Type"] = "text/csv"
                return resp
            elif outputFileType=="json":
                resp = make_response(result.to_json(orient="records"))
                resp.headers["Content-Disposition"] = "attachment; filename=download.json"
                resp.headers["Content-Type"] = "application/json"
                return resp

    except:
        return abort(500)

@app.route("/",methods=["Get"])
def index_view():
    return render_template("index.html",name="Niyaz")
@app.errorhandler(404)
def error_404(e):
    return "The parameters are not correct", 404
@app.errorhandler(500)
def error_500(e):
    return "Something wrong happen Please contact onwer for more detail!", 500


if __name__ == "__main__":
    app.run()