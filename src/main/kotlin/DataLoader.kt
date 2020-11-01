import com.github.doyaaaaaken.kotlincsv.dsl.csvReader
import com.github.doyaaaaaken.kotlincsv.dsl.csvWriter
import java.io.File
import java.util.*

class DataLoader(val filename: String) {
    private var myRandom = Random()

    fun setSeed(seed: Long) {
        myRandom = Random(seed)
    }

    fun readAndSplitTrainTest(targetColIndex: Int, testPercentage: Double): TrainTest {
        val rtable: DataFrame = read()
        val rowsCnt = rtable.rowsCnt()
        val (target, table) = rtable.extractColumn(targetColIndex)

        val res = TrainTest(table.likeC(), target.likeC(), table.likeC(), target.likeC())

        (0 until rowsCnt).forEach {i ->
            if (myRandom.nextInt(10000) < 10000 * testPercentage) {
                res.x_test.appendRow(table[i])
                res.y_test.appendRow(target[i])
            } else {
                res.x_train.appendRow(table[i])
                res.y_train.appendRow(target[i])
            }
        }
        return res
    }

    fun writeCsv(data: DataFrame, labels: List<String>? = null) {
        val rows = data.getRows()
        csvWriter().open(File(filename)) {
            if (labels != null) {
                writeRow(labels)
            }
            writeRows(rows)
        }
    }

    fun writeCsvIndexed(data: DataFrame, labels: List<String>? = null) {
        val rows = data.getRows()
        csvWriter().open(File(filename)) {
            if (labels != null) {
                writeRow(labels)
            }
            rows.forEachIndexed {i, row ->
                val resRow = rows[i].map {it.toString()}.toMutableList()
                resRow.add(i.toString())
                writeRow(resRow)
            }
        }
    }

    fun read(): DataFrame {
        var csv = readCsv()
        if (!csv[0].all {el -> el.toDoubleOrNull() != null}) {
            csv = csv.drop(1)
        }
        return DataFrame(csv)
    }

    private fun readCsv(): List<List<String>> {
        return csvReader().readAll(File(filename))
    }
}

data class TrainTest(val x_train: DataFrame, val y_train: DataFrame, val x_test: DataFrame, val y_test: DataFrame)