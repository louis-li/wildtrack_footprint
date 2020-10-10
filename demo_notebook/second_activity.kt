package com.example.android.plantsapp

import android.annotation.SuppressLint
import android.os.Bundle
import android.widget.Toast
import android.app.Activity
import android.content.Intent
import android.content.pm.ActivityInfo
import android.graphics.*
import android.os.Build
import android.provider.MediaStore
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity;
import android.view.Gravity
import android.graphics.Matrix;
import android.widget.Button
import android.widget.TextView
import kotlinx.android.synthetic.main.second_activity.*;
import java.io.IOException
import java.time.Instant
import java.time.format.DateTimeFormatter
import java.util.*

class second_activity  : AppCompatActivity() {
    private lateinit var mClassifier: Classifier
    private lateinit var mFaceRegressor: EddieFaceRegressor
    private lateinit var mBitmap: Bitmap
    private lateinit var faceBitmap: Bitmap

    private val mCameraRequestCode = 0
    private val mGalleryRequestCode = 2

    private val mInputSize = 299
    private val mModelPath = "output_model.tflite"
    private val faceModelPath="FaceModelFinal.tflite"
    private val mLabelPath = "plant_labels.txt"
    private val mSamplePath = "image1.png"


    @RequiresApi(Build.VERSION_CODES.O)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        setContentView(R.layout.second_activity)
        println("mModelPath: "+mModelPath)
        println("mLabelPath: "+mLabelPath)
        println("mInputSize: "+mInputSize)
        println("assets "+assets)
        //mClassifier = Classifier(assets, mModelPath, mLabelPath, mInputSize)
        //EddieFaceRegressor (assetManager: AssetManager, modelPath: String){
        mFaceRegressor=EddieFaceRegressor(assets,faceModelPath)

        resources.assets.open(mSamplePath).use {
            mBitmap = BitmapFactory.decodeStream(it)
            //faceBitmap=BitmapFactory.decodeStream(it)
            mBitmap = Bitmap.createScaledBitmap(mBitmap, mInputSize, mInputSize, true)
            //fa
            mPhotoImageView.setImageBitmap(mBitmap)
        }

        mCameraButton.setOnClickListener {
            //System.out.println("In mCameraButton.setOnClickListener")
            val callCameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            startActivityForResult(callCameraIntent, mCameraRequestCode)
        }

        mGalleryButton.setOnClickListener {
            //System.out.println("in mGalleryButton.setOnClickListener")
            val callGalleryIntent = Intent(Intent.ACTION_PICK)
            callGalleryIntent.type = "image/*"
            startActivityForResult(callGalleryIntent, mGalleryRequestCode)
        }
        mDetectButton.setOnClickListener {
            //System.out.println("in mDetectButton.setOnClickListener; Need to call recognizeImage")
            val beforeTime= DateTimeFormatter.ISO_INSTANT.format(Instant.now())
            val faceResults=mFaceRegressor.regressImage(mBitmap)
            val afterTime= DateTimeFormatter.ISO_INSTANT.format(Instant.now())
            //val results = mClassifier.recognizeImage(mBitmap).firstOrNull()
            //val results= Array(FloatArray(2))
            //mResultTextView.text= results?.title+"\n Confidence:"+results?.confidence
            mResultTextView.text= "TITLE\n Confidence:"
            System.out.println("Face Results : "+Arrays.toString(faceResults)+"\n")
            val niceFaceResults=faceResults.forEach { Arrays.toString(it)}
            val names=listOf("LeftEye","RightEye","Nose")
            for (x in 0..2){
                val tempName=names[x]
                val tempResult=faceResults[x]
                val x=tempResult[0]
                val y=tempResult[1]
                System.out.println(tempName+" : "+x+","+y+"\n")
            }
            System.out.println("Time before call to regress "+beforeTime)
            System.out.println("Time after call to regress "+afterTime)
            //System.out.println("Face results : "+faceResults[0].toString())
            var betterBM=addOntoBitMap(faceResults[0][0],faceResults[0][1],mBitmap)
            betterBM=addOntoBitMap(faceResults[1][0],faceResults[1][1],betterBM)
            betterBM=addOntoBitMap(faceResults[2][0],faceResults[2][1],betterBM)
            mPhotoImageView.setImageBitmap(betterBM)

        }

        moreInfoText.setOnClickListener{
            //System.out.println("in moreInfoText.setOnClickListener")
            Intent(applicationContext, InfoActivity::class.java).also {
                startActivity(it)
            }
        }
    }

    @SuppressLint("MissingSuperCall")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        if(requestCode == mCameraRequestCode){
            //Considérons le cas de la caméra annulée
            if(resultCode == Activity.RESULT_OK && data != null) {
                mBitmap = data.extras!!.get("data") as Bitmap
                mBitmap = scaleImage(mBitmap)
                val toast = Toast.makeText(this, ("Image crop to: w= ${mBitmap.width} h= ${mBitmap.height}"), Toast.LENGTH_LONG)
                toast.setGravity(Gravity.BOTTOM, 0, 20)
                toast.show()
                mPhotoImageView.setImageBitmap(mBitmap)
                mResultTextView.text= "Your photo image set now."
            } else {
                Toast.makeText(this, "Camera cancel..", Toast.LENGTH_LONG).show()
            }
        } else if(requestCode == mGalleryRequestCode) {
            if (data != null) {
                val uri = data.data

                try {
                    mBitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
                } catch (e: IOException) {
                    e.printStackTrace()
                }

                println("Success!!!")
                mBitmap = scaleImage(mBitmap)
                mPhotoImageView.setImageBitmap(mBitmap)

            }
        } else {
            Toast.makeText(this, "Unrecognized request code", Toast.LENGTH_LONG).show()

        }

    }

    fun addOntoBitMap(face_x:Float,face_y:Float,tb:Bitmap) : Bitmap
    {
        //inverseScale(pred_x:Float,pred_y:Float,b_width:Int,b_height:Int ) : FloatArray
        val proper_scale_coor=mFaceRegressor.inverseScale(face_x,face_y,tb.width,tb.height)
        val theCanvas=Canvas(tb)
        val myPaint =
            Paint().apply {
                isAntiAlias = true
                color = Color.GREEN
                style = Paint.Style.STROKE
            }
        //theCanvas.drawPoint(proper_scale_coor[0],proper_scale_coor[1],myPaint)
        val proper_x=proper_scale_coor[0].toInt()
        val proper_y=proper_scale_coor[1].toInt()
        val rect_dim=5
        val myRectangle=Rect(proper_x,proper_y,proper_x+rect_dim,proper_y+rect_dim)
        theCanvas.drawRect(myRectangle,myPaint)
        val identityMatrix=Matrix()
        //val resultBitmap=theCanvas.drawBitmap(tb,identityMatrix,null)
        return Bitmap.createBitmap(tb)
    }


    fun scaleImage(bitmap: Bitmap?): Bitmap {
        val orignalWidth = bitmap!!.width
        val originalHeight = bitmap.height
        val scaleWidth = mInputSize.toFloat() / orignalWidth
        val scaleHeight = mInputSize.toFloat() / originalHeight
        val matrix = Matrix()
        matrix.postScale(scaleWidth, scaleHeight)
        return Bitmap.createBitmap(bitmap, 0, 0, orignalWidth, originalHeight, matrix, true)
    }

}
