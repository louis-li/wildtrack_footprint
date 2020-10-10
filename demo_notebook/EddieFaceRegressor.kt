package com.example.android.plantsapp

import android.content.res.AssetManager
import android.graphics.*
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.io.FileNotFoundException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*

/*
left_eye_center
right_eye_center
left_eye_inner_corner
left_eye_outer_corner
right_eye_inner_corner
right_eye_outer_corner
left_eyebrow_inner_end
left_eyebrow_outer_end
right_eyebrow_inner_end
right_eyebrow_outer_end
nose_tip
mouth_left_corner
mouth_right_corner
mouth_center_top_lip
mouth_center_bottom_lip
 */

class EddieFaceRegressor (assetManager: AssetManager, modelPath: String){


        private var INTERPRETER: Interpreter
    //private var LABEL_LIST: List<String>
    private val INPUT_SIZE: Int = 96 //per Louis
    private val PIXEL_SIZE: Int = 1 //greyscale  ; used to be 3
    private val IMAGE_MEAN = 0
    private val IMAGE_STD = 255.0f
    //private val MAX_RESULTS = 3
    //private val THRESHOLD = 0.7f

    init {
        INTERPRETER = Interpreter(loadModelFile(assetManager, modelPath))
        //LABEL_LIST = loadLabelList(assetManager, labelPath)
    }

    public fun inverseScale(pred_x:Float,pred_y:Float,b_width:Int,b_height:Int ) : FloatArray
    {
        val myArray=FloatArray(2)
        val ratio_x=pred_x/INPUT_SIZE
        val ratio_y=pred_y/INPUT_SIZE
        val out_x=ratio_x*b_width
        val out_y=ratio_y*b_height
        myArray[0]=out_x
        myArray[1]=out_y
        return myArray
    }

/*    public Bitmap toGrayscale(Bitmap bmpOriginal)
{
    int width, height;
    height = bmpOriginal.getHeight();
    width = bmpOriginal.getWidth();

    Bitmap bmpGrayscale = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
    Canvas c = new Canvas(bmpGrayscale);
    Paint paint = new Paint();
    ColorMatrix cm = new ColorMatrix();
    cm.setSaturation(0);
    ColorMatrixColorFilter f = new ColorMatrixColorFilter(cm);
    paint.setColorFilter(f);
    c.drawBitmap(bmpOriginal, 0, 0, paint);
    return bmpGrayscale;
}*/
    private fun toGrayscale(bmpOriginal:Bitmap): Bitmap {
//https://stackoverflow.com/questions/3373860/convert-a-bitmap-to-grayscale-in-android

    var width=0
    var height=0
    height = bmpOriginal.getHeight();
    width = bmpOriginal.getWidth();

    val bmpGrayscale = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
    val mycanvas = Canvas(bmpGrayscale)
    val paint = Paint()
    val cm = ColorMatrix()
    val myZeroFloat=0.0f
    cm.setSaturation(myZeroFloat)
    val f = ColorMatrixColorFilter(cm)
    paint.setColorFilter(f)
    mycanvas.drawBitmap(bmpOriginal, myZeroFloat, myZeroFloat, paint)
    return bmpGrayscale
}

    private fun loadModelFile(assetManager: AssetManager, modelPath: String): MappedByteBuffer {
        System.out.println("To LOAD model from "+modelPath)
        val fileDescriptor = assetManager.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        System.out.println("Declared length is "+declaredLength)
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)

    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE)
        byteBuffer.order(ByteOrder.nativeOrder())
        val intValues = IntArray(INPUT_SIZE * INPUT_SIZE)

        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        var pixel = 0
        for (i in 0 until INPUT_SIZE) {
            for (j in 0 until INPUT_SIZE) {
                val `val` = intValues[pixel++]

                //byteBuffer.putFloat((((`val`.shr(16)  and 0xFF) - IMAGE_MEAN) / IMAGE_STD)) R?
                //byteBuffer.putFloat((((`val`.shr(8) and 0xFF) - IMAGE_MEAN) / IMAGE_STD)) G?
                byteBuffer.putFloat((((`val` and 0xFF) - IMAGE_MEAN) / IMAGE_STD))
            }
        }
        return byteBuffer
    }


    fun regressImage(bitmap: Bitmap): Array<FloatArray>  {
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false)
        val grayScaledBitmap=toGrayscale(scaledBitmap)
        val byteBuffer = convertBitmapToByteBuffer(grayScaledBitmap)

/*
left_eye_center 0***************
right_eye_center 1*************************
left_eye_inner_corner
left_eye_outer_corner
right_eye_inner_corner 4
right_eye_outer_corner
left_eyebrow_inner_end
left_eyebrow_outer_end
right_eyebrow_inner_end
right_eyebrow_outer_end
nose_tip 10**************
mouth_left_corner
mouth_right_corner
mouth_center_top_lip
mouth_center_bottom_lip
 */
        val leftEyeIdx=0
        val rightEyeIdx=1
        val noseIdx=10
        /*var result=arrayOf(
            arrayOf(0.0f, 0.0f), //0th (first) (left eye)
            arrayOf(0.0f, 0.0f), // right eye
            arrayOf(0.0f, 0.0f),
            arrayOf(0.0f, 0.0f),
            arrayOf(0.0f, 0.0f), //4th (fifth)
            arrayOf(0.0f, 0.0f),
            arrayOf(0.0f, 0.0f),
            arrayOf(0.0f, 0.0f),
            arrayOf(0.0f, 0.0f),
            arrayOf(0.0f, 0.0f), //9th (tenth)
            arrayOf(0.0f, 0.0f), //10 (nose)
            arrayOf(0.0f, 0.0f),
            arrayOf(0.0f, 0.0f),
            arrayOf(0.0f, 0.0f),
            arrayOf(0.0f, 0.0f))*/
        val result = Array(1) { FloatArray(30) }
        INTERPRETER.run(byteBuffer, result)
        var left_eye_result=FloatArray(2)
        left_eye_result[0]=result[0][leftEyeIdx*2]
        left_eye_result[1]=result[0][(leftEyeIdx*2)+1]
        var right_eye_result=FloatArray(2)
        right_eye_result[0]=result[0][rightEyeIdx*2]
        right_eye_result[1]=result[0][(rightEyeIdx*2)+1]
        var nose_result=FloatArray(2)
        nose_result[0]=result[0][noseIdx*2]
        nose_result[1]=result[0][(noseIdx*2)+1]
        val points_result=arrayOf(left_eye_result,right_eye_result,nose_result)
        return points_result
    }


}