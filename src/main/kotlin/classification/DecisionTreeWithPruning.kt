package classification

import DataFrame
import java.nio.file.NotDirectoryException
import java.util.*
import kotlin.math.abs
import kotlin.math.round

class DecisionTreeWithPruning(
        private var trainPart: Double,
        maxDepth: Int,
        minNodeData: Int,
        parametersToUse: List<Int>? = null,
        uncertantyFunction: Uncertanty = Uncertanty.ENT
) : DecisionTreeClassifier(maxDepth, minNodeData, parametersToUse, uncertantyFunction) {
    private var myRandom = Random()

    fun setSeed(seed: Long) {myRandom = Random(seed)}

    override fun fit(table: DataFrame, target: DataFrame) {
        val trainData = table.likeC()
        val testData = table.likeC()
        val trainTarget = target.likeC()
        val testTarget = target.likeC()
        table.getRows().forEachIndexed {i, row ->
            if (myRandom.nextInt(10000) < trainPart * 10000) {
                trainData.appendRow(row)
                trainTarget.appendRow(target[i])
            } else {
                testData.appendRow(row)
                testTarget.appendRow(target[i])
            }
        }

        super.fit(trainData, trainTarget)

        prune(myRoot!!, testData, testTarget)
    }

    private fun prune(node: Node, data: DataFrame, target: DataFrame): Node {
        if (node.isTerminal) {
            return node
        }
        if (data.rowsCnt() <= 3) {
            return getLargestNode(node)
        }
        val ldata = data.likeC()
        val ltarget = target.likeC()
        val rdata = data.likeC()
        val rtarget = target.likeC()

        (0 until data.rowsCnt()).forEach {i ->
            val v = data[i][node.featureIndex!!]

            if (!v.isNaN()) {
                if (v <= node.bound!!) {
                    ldata.appendRow(data[i])
                    ltarget.appendRow(target[i])
                } else {
                    rdata.appendRow(data[i])
                    rtarget.appendRow(target[i])
                }
            }
        }

        val newL = prune(node.left!!, ldata, ltarget)
        val newR = prune(node.right!!, rdata, rtarget)

        node.left = newL
        node.right = newR

        val node1 = Node()
        node1.isTerminal = true
        node1.ans = 1.0

        val node0 = Node()
        node0.isTerminal = true
        node0.ans = 0.0

        val nodeVars = listOf(node, node.left!!, node.right!!, node1, node0)

        var optAns = 1e9
        var optNode: Node? = null

        nodeVars.forEach {newNode ->
            val cans = getErrWithNode(newNode, data, target)
            if (cans < optAns) {
                optAns = cans
                optNode = newNode
            }
        }

        return optNode!!
    }

    private fun getErrWithNode(node: Node, table: DataFrame, target: DataFrame): Double {
        var res = 0.0
        table.getRows().forEachIndexed{i, row ->
            val cans = getAns(node, row)
            res += abs(cans - target[i][0])
        }

        return res
    }

    private fun getLargestNode(node: Node): Node {
        if (node.isTerminal) {
            return node
        }

        if (node.leftCoef!! > 0.5) {
            return getLargestNode(node.left!!)
        } else {
            return getLargestNode(node.right!!)
        }
    }
}