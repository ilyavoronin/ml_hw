import java.lang.Integer.min

class CrossValidation : ModelEvaluator<List<Double>> {
    private var myK: Int = 0

    fun withK(k: Int): CrossValidation {
        myK = k
        return this
    }
    override fun getQuality(model: RegressionModel, data: DataFrame, target: DataFrame): List<Double> {
        if (myK == 0) {
            myK = data.rowsCnt()
        }

        val ress = mutableListOf<Double>()

        val (shuffledData, shuffledTarget) = data.shuffleWith(target)

        val testSize = data.rowsCnt() / myK

        for (i in 0 until shuffledData.rowsCnt() step testSize) {
            println(i)
            val l = i
            val r = min(shuffledData.rowsCnt(), i + testSize)

            val res = shuffledData.extractWith(shuffledTarget, l, r)

            val curRes = runModel(model, res.part1, res.part1other, res.part2, res.part2other)
            ress.add(curRes)
        }
        return ress
    }

    private fun runModel(model: RegressionModel, data: DataFrame, target: DataFrame, testData: DataFrame, testTarget: DataFrame): Double {
        model.fit(data, target)

        val res = model.predict(testData)

        return meanSqError(res, testTarget)
    }

}