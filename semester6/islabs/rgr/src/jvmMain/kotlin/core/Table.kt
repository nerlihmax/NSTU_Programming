package core

import androidx.compose.desktop.ui.tooling.preview.Preview
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.material.MaterialTheme
import androidx.compose.material.Surface
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp

@Composable
fun Table(
    data: TableData,
    onEditing: (Int) -> Unit = {},
    onDeleting: (Int) -> Unit = {},
    onShowInfo: (Int) -> Unit = {},
    infoRequired: Boolean = false,
    modifier: Modifier = Modifier,
) {
    val idWeight = 0.1f
    val colWeight = 0.8f
    BoxWithConstraints(modifier) {
        Column(
            modifier = Modifier.fillMaxSize()
        ) {
            // header
            Row(modifier = Modifier.weight(1f)) {
                repeat(data.header.size) { idx ->
                    TableCell(data.header[idx], if (idx == 0) idWeight else colWeight)
                }
                Spacer(modifier = Modifier.weight(0.08f))
                Spacer(modifier = Modifier.weight(0.08f))
                if (infoRequired) Spacer(modifier = Modifier.weight(0.08f))
            }

            data.data.forEach {
                val (row) = it
                Row(modifier = Modifier.weight(1f), verticalAlignment = Alignment.CenterVertically) {
                    repeat(row.size) { colIdx ->
                        TableCell(
                            text = row[colIdx],
                            weight = if (colIdx == 0) idWeight else colWeight,
                            type = CellType.DATA,
                        )
                    }

                    TableCell(
                        type = CellType.EDIT,
                        weight = 0.08f,
                        modifier = Modifier.widthIn(min = 50.dp),
                        onClick = {
                            onEditing(row[0].toInt())
                        },
                    )
                    TableCell(
                        type = CellType.DELETE,
                        weight = 0.08f,
                        modifier = Modifier.widthIn(50.dp),
                        onClick = {
                            onDeleting(row[0].toInt())
                        },
                    )
                    if (infoRequired) {
                        TableCell(
                            type = CellType.INFO,
                            weight = 0.08f,
                            modifier = Modifier.widthIn(50.dp),
                            onClick = {
                                onShowInfo(row[0].toInt())
                            },
                        )
                    }
                }
            }
        }
    }
}

@Preview
@Composable
fun TablePreview() {
    MaterialTheme {
        Surface(modifier = Modifier.background(Color.White)) {
            Table(
                TableData(
                    header = listOf("id", "data1", "data2"), data = listOf(
                        DataRow(listOf("1", "2", "3")),
                        DataRow(listOf("1", "2", "3")),
                        DataRow(listOf("1", "2", "3")),
                        DataRow(listOf("1", "2", "3")),
                    )
                )
            )
        }
    }
}