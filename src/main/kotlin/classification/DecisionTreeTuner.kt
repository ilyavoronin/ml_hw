package classification

import DataFrame
import aucRoc
import java.util.*

class DecisionTreeTuner(
    private var trainPart: Double = 0.6,
    private var maxDepth: Int = 9,
    private var minNodeData: Int = 4,
    private var parametersToUse: List<Int>? = null,
    private var uncertantyFunction: DecisionTreeClassifier.Uncertanty = DecisionTreeClassifier.Uncertanty.ENT
) {
    private val myRandom = Random()

    fun getBest(treesCnt: Int, data: DataFrame, target: DataFrame): DecisionTreeClassifier {
        val myTrees = mutableListOf<DecisionTreeClassifier>()

        (0 until treesCnt).forEach {
            if (trainPart != 1.0) {
                val newTree = DecisionTreeWithPruning(trainPart, maxDepth + getDiff(), minNodeData + getDiff(), parametersToUse, uncertantyFunction)
                newTree.setSeed(myRandom.nextLong())
                myTrees.add(newTree)
            } else {
                myTrees.add(DecisionTreeClassifier(maxDepth + getDiff(), minNodeData + getDiff(), parametersToUse, uncertantyFunction))
            }
        }
        return getBest(data, target, *myTrees.toTypedArray())
    }

    fun getBest(data:DataFrame, target: DataFrame, vararg trees: DecisionTreeClassifier): DecisionTreeClassifier {
        var bestRes = 0.0
        var bestModel: DecisionTreeClassifier? = null

        trees.forEachIndexed {i, tree ->
            println("TreeTuner $i out of ${trees.size}")
            val cv = AggregatingCrossValidation(tree, ::aucRoc)
            cv.setK(10)

            val currQ = cv.getQuality(data, target)

            print(" $currQ ")

            if (currQ > bestRes) {
                bestRes = currQ
                bestModel = tree
            }
        }
        println(bestRes)

        return bestModel!!
    }

    private fun getDiff(): Int {
        return myRandom.nextInt(3)
    }
}