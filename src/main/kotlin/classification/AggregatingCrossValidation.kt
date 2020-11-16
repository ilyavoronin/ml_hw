package classification

import DataFrame
import ModelEvaluator
import MLModel

class AggregatingCrossValidation(
        override val model: MLModel,
        val aggregator: (expected: DataFrame, actual: DataFrame) -> Double
) : ModelEvaluator<Double> {

    private var myK: Int = -1
    private var mySeed: Long? = null

    fun setK(k: Int) {myK = k}

    fun setSeed(seed: Long) { mySeed = seed }

    override fun getQuality(data: DataFrame, target: DataFrame): Double {

        val ress = DataFrame(0, 1)
        val (shuffledData, shuffledTarget) = data.shuffleWith(target, mySeed)

        val testSize = if (myK != -1) data.rowsCnt() / myK else 1

        for (i in 0 until shuffledData.rowsCnt() step testSize) {
            print(i.toString() + " ")
            val l = i
            val r = Integer.min(shuffledData.rowsCnt(), i + testSize)

            val res = shuffledData.extractWith(shuffledTarget, l, r)

            val curRes = runModel(model, res.part1, res.part1other, res.part2)

            curRes.forEach {ress.appendRow(listOf(it))}
        }
        return aggregator(shuffledTarget, ress)
    }

    override fun getDoubleQuality(data: DataFrame, target: DataFrame): Double {
        return getQuality(data, target)
    }

    private fun runModel(model: MLModel, data: DataFrame, target: DataFrame, testData: DataFrame): List<Double> {
        model.fit(data, target)

        val res = model.predict(testData)

        return res.tr()[0]
    }

}