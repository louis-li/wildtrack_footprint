package com.example.android.plantsapp

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*
import java.lang.Math

class SpeciesClassifier (assetManager: AssetManager, modelPath: String, labelPath: String, inputSize: Int,mName:String){

    private var MODEL_NAME: String
    private var INTERPRETER: Interpreter
    private var LABEL_LIST: List<String>
    private val INPUT_SIZE: Int = inputSize
    private val PIXEL_SIZE: Int = 3
    private val IMAGE_MEAN = 0
    private val IMAGE_STD = 1.0f
    private val MAX_RESULTS = 3
    private val THRESHOLD = 0.7f



    init {
        MODEL_NAME=mName
        INTERPRETER = Interpreter(loadModelFile(assetManager, modelPath))
        LABEL_LIST = loadLabelList(assetManager, labelPath)
        System.out.println("Loading model file "+modelPath+" for model "+MODEL_NAME)
        for(l in 0 until LABEL_LIST.size)
            {
                System.out.println("Model "+MODEL_NAME+", label "+l+" is "+LABEL_LIST[l])
            }
    }

    public fun getName():String{
        return this.MODEL_NAME
    }

    private fun loadModelFile(assetManager: AssetManager, modelPath: String): MappedByteBuffer {
        val fileDescriptor = assetManager.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun loadLabelList(assetManager: AssetManager, labelPath: String): List<String> {
        return assetManager.open(labelPath).bufferedReader().useLines { it.toList() }

    }

    @Synchronized
    fun  recognizeImage(bitmap: Bitmap): String {
        //System.out.println("INPUT_SIZE is "+INPUT_SIZE)
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false)
        val byteBuffer = convertBitmapToByteBuffer(scaledBitmap)
        val result = Array(1) { FloatArray(LABEL_LIST.size) }
        //val inv_logits = Array(1) { FloatArray(LABEL_LIST.size) }
        //System.out.println("Direct results array before call "+Arrays.toString(result[0]))
        INTERPRETER.run(byteBuffer, result)
        System.out.println("Model "+MODEL_NAME+": Direct results : "+Arrays.toString(result[0]))
        val maxIdx = result[0].indexOf(result[0].max()?:-99999.999999f)
        //System.out.println("MaxIdx is "+maxIdx)
        //System.out.println("Max val is  is "+result[0][maxIdx])
        val label=LABEL_LIST[maxIdx]
        System.out.println("Model : "+MODEL_NAME+" Predicted Label is "+label)
        return label
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

                /*byteBuffer.putFloat((((`val`.shr(16)  and 0xFF) - IMAGE_MEAN) / IMAGE_STD))
                byteBuffer.putFloat((((`val`.shr(8) and 0xFF) - IMAGE_MEAN) / IMAGE_STD))
                byteBuffer.putFloat((((`val` and 0xFF) - IMAGE_MEAN) / IMAGE_STD))*/
                byteBuffer.putFloat((((`val`.shr(16)  and 0xFF) - IMAGE_MEAN) / IMAGE_STD))
                byteBuffer.putFloat((((`val`.shr(8) and 0xFF) - IMAGE_MEAN) / IMAGE_STD))
                byteBuffer.putFloat((((`val` and 0xFF) - IMAGE_MEAN) / IMAGE_STD))

            }

        }
        return byteBuffer
    }



}













