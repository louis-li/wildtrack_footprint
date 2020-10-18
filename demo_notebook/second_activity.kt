package com.example.android.plantsapp

import android.annotation.SuppressLint
import android.app.Activity
import android.content.Intent
import android.content.pm.ActivityInfo
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.text.TextUtils
import android.view.Gravity
import android.widget.Toast
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import kotlinx.android.synthetic.main.second_activity.*
import java.io.*
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import java.time.Instant
import java.time.format.DateTimeFormatter


class second_activity  : AppCompatActivity() {
    private lateinit var mClassifier: Classifier
    private lateinit var species_classifier:SpeciesClassifier
    private lateinit var mBitmap: Bitmap
    private lateinit var myVotes:ArrayList<String>

    private val mCameraRequestCode = 0
    private val mGalleryRequestCode = 2

    private val mInputSize = 299
    private val mModelPath = "output_model.tflite"
    private val sModelPath="xception_tf2.tflite"
    private val sLabelPath="exception_labels.ready.txt"
    private val sInputSize=224
    private val mLabelPath = "plant_labels.txt"
    private val mSamplePath = "image1.png"


    //models for list
    private lateinit var exception_tflite_Louis:SpeciesClassifier
    private lateinit var inception_tflite_Louis:SpeciesClassifier
    private lateinit var nasnet_tflite_louis:SpeciesClassifier
    private lateinit var model_list:ArrayList<SpeciesClassifier>


    private fun sendPredictionResult(p_url: String, kv_pairs: HashMap<String, String>)
    {

        try {
            var url=p_url
            if(kv_pairs.isNotEmpty())
                {
                url+="?"
                var pairs= mutableListOf<String>()
                //val mutableList = mutableListOf<Kolory>()
                //val mutableList : MutableList<Kolory> = arrayListOf()
                for((k,v) in kv_pairs)
                    {
                    val new_pair=k+"="+URLEncoder.encode(v,"UTF-8")
                    //System.out.println("new pair is "+new_pair)
                    pairs.add(new_pair)
                    }
                var req_kvp=TextUtils.join("&",pairs)
                url+=req_kvp
                }
            System.out.println("Final URL for submission is "+url)
            readURLAsText(url)

            }
        catch (e: Exception)
        {
            e.printStackTrace()
        }
    }


    //https://stackoverflow.com/questions/8992964/android-load-from-url-to-bitmap
    //key is "decodestream"
    fun getBitmapFromURL(src: String?): Bitmap? {
        return try {
            val url = URL(src)
            val connection = url.openConnection() as HttpURLConnection
            connection.doInput = true
            connection.connect()
            val input: InputStream = connection.inputStream
            val to_ret=BitmapFactory.decodeStream(input)
            return to_ret
        } catch (e: IOException) {
            // Log exception
            null
        }
    }



    private fun readURLAsText(urls: String):  ArrayList<String>   {
            //val url = URL("http://192.168.1.76:8000/img_list.txt")
            val url = URL(urls)
            val con: HttpURLConnection = url.openConnection() as HttpURLConnection
            val myisreader=InputStreamReader(con.inputStream)
            val mybufferedreader=BufferedReader(myisreader)
            var inputLine=mybufferedreader.readLine()
            val myLines=ArrayList<String>()
            while(inputLine!=null)
                {
                myLines.add(inputLine)
                inputLine=mybufferedreader.readLine()
                }
            myisreader.close()
            return myLines
        }

    fun launchThread() {
        Thread(Runnable {
            // a potentially time consuming task
            val run_id=DateTimeFormatter.ISO_INSTANT.format(Instant.now())
            System.out.println("A Thread is started")
            val url_base = "http://192.168.1.76:8000/"
            var img_fof_url = url_base + "img_list_louis_test.txt"
            var list_of_files = readURLAsText(img_fof_url)
            //System.out.println("List : "+list_of_files.toString())
            //val myurl_encoder=URLEncoder()
            System.out.println("Downloaded a list of " + list_of_files.size + " files!")
            for (f in 0 until list_of_files.size) {
            //for (f in 0 until 10) {
                //for (f in 0 until 5) {
                System.out.println("File " + (f+1) + " of "+list_of_files.size+": " + list_of_files[f])
                val img_url = url_base + list_of_files[f]
                System.out.println("Image URL is " + img_url)
                var encoded_img_path = list_of_files[f]
                //var encoded_img_path=URLEncoder.encode(list_of_files[f],"UTF-8")
                //encoded_img_path=encoded_img_path.replace("%2F","/")

                val img_url_encoded = url_base + encoded_img_path
                System.out.println("Encoded Image URL is " + img_url_encoded)
                val tempBitmap = getBitmapFromURL(img_url_encoded)
                if (tempBitmap != null) {
                    for (m in model_list) {
                        val beforeTime = DateTimeFormatter.ISO_INSTANT.format(Instant.now())
                        val temp_result = m.recognizeImage(tempBitmap)
                        val afterTime = DateTimeFormatter.ISO_INSTANT.format(Instant.now())
                        System.out.println("From url " + img_url + ", model " + m.getName() + " result is " + temp_result)
                        System.out.println("Times before/after prediction : " + beforeTime + "/" + afterTime)
                        var infoMap=HashMap<String,String>()
                        infoMap.put("run_id",run_id)
                        infoMap.put("file",list_of_files[f])
                        infoMap.put("file_id",""+f+"")
                        infoMap.put("result",temp_result)
                        infoMap.put("model_name",m.getName())
                        infoMap.put("start_time",beforeTime)
                        infoMap.put("end_time",afterTime)
                        val send_req_url="http://192.168.1.76:5000/report_image"
                        sendPredictionResult(send_req_url,infoMap)

                    }
                } else {
                    System.out.println("Bitmap from " + img_url_encoded + " was null!")
                }

            }


        }).start()
    }




    @RequiresApi(Build.VERSION_CODES.O)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        setContentView(R.layout.second_activity)
        //mClassifier = Classifier(assets, mModelPath, mLabelPath, mInputSize)
        //species_classifier=SpeciesClassifier(assets,sModelPath,sLabelPath,sInputSize,"exception_tf2_model")
        exception_tflite_Louis=SpeciesClassifier(
            assets,
            "xception.tflite",
            "species_13_labels.txt",
            224,
            "exception_louis_direct"
        )
        inception_tflite_Louis=SpeciesClassifier(
            assets,
            "InceptionResNetV2.tflite",
            "species_13_labels.txt",
            299,
            "Inception_louis_direct"
        )
        nasnet_tflite_louis=SpeciesClassifier(
            assets,
            "NASNetLarge.tflite",
            "species_13_labels.txt",
            331,
            "nasnet_louis_direct"
        )
        model_list= ArrayList()
        myVotes= ArrayList()
        model_list.add(exception_tflite_Louis)
        model_list.add(inception_tflite_Louis)
        model_list.add(nasnet_tflite_louis)


        //launchThread()

        /*resources.assets.open(mSamplePath).use {
            mBitmap = BitmapFactory.decodeStream(it)
            mBitmap = Bitmap.createScaledBitmap(mBitmap, mInputSize, mInputSize, true)
            mPhotoImageView.setImageBitmap(mBitmap)
        }*/

        mCameraButton.setOnClickListener {
            val callCameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            startActivityForResult(callCameraIntent, mCameraRequestCode)
        }

        mGalleryButton.setOnClickListener {
            val callGalleryIntent = Intent(Intent.ACTION_PICK)
            callGalleryIntent.type = "image/*"
            startActivityForResult(callGalleryIntent, mGalleryRequestCode)
        }
        mDetectButton.setOnClickListener {
            var slabel=""
            myVotes.removeAll(myVotes)
            for(m in model_list)
                {
                    System.out.println("To invoke model " + m.getName())

                    val beforeTime= DateTimeFormatter.ISO_INSTANT.format(Instant.now())
                    slabel=m.recognizeImage(mBitmap)
                    val afterTime= DateTimeFormatter.ISO_INSTANT.format(Instant.now())
                    System.out.println("Time before call " + beforeTime)
                    System.out.println("Time after call " + afterTime)
                    myVotes.add(slabel)
                }
            System.out.println("Votes are : " + myVotes.toString())
            slabel=mostFrequent(myVotes)
            //mResultTextView.text= results?.title+"\n Confidence:"+results?.confidence
            mResultTextView.text= slabel
        }

        /*moreInfoText.setOnClickListener{
            Intent(applicationContext, InfoActivity::class.java).also {
                startActivity(it)
            }
        }*/
    }


    private fun mostFrequent(ml: ArrayList<String>):String
        {
            if(ml.isEmpty())
                {
                return "ERROR, empty list"
                }
            //https://stackoverflow.com/questions/48530786/kotlin-find-most-common-element-in-collection
            val numbersByElement = ml.groupingBy { it }.eachCount()
            var mostCommon= numbersByElement.maxBy { it.value }!!.key // gives an Int?
            return mostCommon
            //gives something like this {1=3, 2=5, 3=4, 5=2, 4=1}
        }


    @SuppressLint("MissingSuperCall")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        if(requestCode == mCameraRequestCode){
            //Considérons le cas de la caméra annulée
            if(resultCode == Activity.RESULT_OK && data != null) {
                mBitmap = data.extras!!.get("data") as Bitmap
                mBitmap = scaleImage(mBitmap)
                val toast = Toast.makeText(
                    this,
                    ("Image crop to: w= ${mBitmap.width} h= ${mBitmap.height}"),
                    Toast.LENGTH_LONG
                )
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
                    //System.out.println("Setting mbitmap with uri="+uri)
                    //System.out.println("content resolver "+this.contentResolver)
                    mBitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
                } catch (e: IOException) {
                    e.printStackTrace()
                    println("Error, see stack trace")
                }

                //println("Just before scaleimage not called!!!")
                //mBitmap = scaleImage(mBitmap)
                mPhotoImageView.setImageBitmap(mBitmap)

            }
        } else {
            Toast.makeText(this, "Unrecognized request code", Toast.LENGTH_LONG).show()

        }

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
