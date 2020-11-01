import java.lang.IllegalStateException

class DataFrame {
    private var myTable: MutableList<MutableList<Double>> = mutableListOf()
    private var myRowsCnt: Int
    private var myColsCnt: Int
    constructor(table: List<List<String>>) {
        table.forEach {row -> myTable.add(row.map {el -> el.toDouble()}.toMutableList())}
        myRowsCnt = myTable.size
        myColsCnt = myTable[0].size
    }

    constructor(rowsCnt: Int, colsCnt: Int) {
        myRowsCnt = rowsCnt
        myColsCnt = colsCnt
    }

    fun rowsCnt() = myRowsCnt

    fun colsCnt() = myColsCnt

    fun getRows() = this.copy().myTable

    operator fun get(rowIndex: Int, colIndex: Int): Double {
        return myTable[rowIndex][colIndex]
    }

    operator fun get(rowIndex: Int): List<Double> {
        return myTable[rowIndex]
    }

    fun getCol(colIndex: Int): DataFrame {
        val res = DataFrame(myRowsCnt, 1)
        (0 until myRowsCnt).forEach { i -> res.appendRow(listOf(myTable[i][colIndex]))}
        return res
    }

    fun appendRow(row: List<Double>) {
        if (row.size != myColsCnt) {
            throw IllegalStateException("wrong y dimension")
        }
        myTable.add(row.toMutableList())
        myRowsCnt += 1
    }

    fun appendColumn(col: List<Double>) {
        if (col.size != myRowsCnt) {
            throw IllegalStateException("wrong x dimension")
        }
        myTable.forEachIndexed {i, row ->
            row.add(col[i])
        }
        myColsCnt += 1
    }

    fun joinColumns(other: DataFrame): DataFrame {
        val copy = this.copy()
        other.tr().getRows().forEach {row -> copy.appendColumn(row)}
        return copy
    }

    fun splitColsBy(i: Int): Pair<DataFrame, DataFrame> {
        val res1 = DataFrame(0, i)
        val res2 = DataFrame(0, colsCnt() - i)
        myTable.forEach {row ->
            val row1 = mutableListOf<Double>()
            val row2 = mutableListOf<Double>()
            (0 until i).forEach {j -> row1.add(row[j])}
            (i until colsCnt()).forEach {j -> row2.add(row[j])}
            res1.appendRow(row1)
            res2.appendRow(row2)
        }
        return Pair(res1, res2)
    }

    fun shuffleWith(other: DataFrame): Pair<DataFrame, DataFrame> {
        val thisColCnt = colsCnt()
        val un = this.joinColumns(other)
        un.myTable.shuffle()
        return un.splitColsBy(thisColCnt)
    }

    data class SplitRes(val part1: DataFrame, val part1other: DataFrame, val part2: DataFrame, val part2other: DataFrame)

    fun extractWith(other: DataFrame, from: Int, to: Int): SplitRes {
        val res = SplitRes(likeC(), other.likeC(), likeC(), other.likeC())
        (0 until rowsCnt()).forEach {i ->
            if (i < to && i >= from) {
                res.part2.appendRow(this[i])
                res.part2other.appendRow(other[i])
            } else {
                res.part1.appendRow(this[i])
                res.part1other.appendRow(other[i])
            }
        }
        return res
    }

    fun likeC(): DataFrame {
        return DataFrame(0, myColsCnt)
    }

    fun tr(): DataFrame {
        val res = DataFrame(0, myRowsCnt)
        for (j in 0 until myColsCnt) {
            val newRow = mutableListOf<Double>()
            for (i in 0 until myRowsCnt) {
                newRow.add(myTable[i][j])
            }
            res.appendRow(newRow)
        }
        return res
    }

    fun sortBy(j: Int): DataFrame {
        val res = this.copy()
        res.myTable.sortBy { it[j] }
        return res
    }

    fun med(j: Int): Double {
        val tmp = this.sortBy(j)
        if (rowsCnt() % 2 == 0) {
            return (tmp[rowsCnt() / 2 - 1][j] + tmp[rowsCnt() / 2 + 1][j]) / 2
        } else {
            return tmp[rowsCnt() / 2][j]
        }
    }

    fun splitBy(j: Int, value: Double): Pair<DataFrame, DataFrame> {
        val left = likeC()
        val right = likeC()

        myTable.forEach {row ->
            if (row[j] <= value) {
                left.appendRow(row)
            } else {
                right.appendRow(row)
            }
        }
        return Pair(left, right)
    }

    fun copy(): DataFrame {
        val res = DataFrame(0, myColsCnt)
        myTable.forEach {row -> res.appendRow(row.toList())}
        return res
    }

    fun extractColumn(i : Int): Pair<DataFrame, DataFrame> {
        val column = DataFrame(myRowsCnt, 1)
        val r = this.copy()
        (0 until myRowsCnt).forEach { j ->
            column.appendRow(listOf(r.myTable[j].removeAt(i)))
        }
        r.myColsCnt -= 1
        return Pair(column, r)
    }
}