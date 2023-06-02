package core

import androidx.compose.foundation.ScrollState
import androidx.compose.foundation.border
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.wrapContentHeight
import androidx.compose.material.Icon
import androidx.compose.material.IconButton
import androidx.compose.material.Text
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.Info
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import core.CellType.ACCEPT
import core.CellType.CANCEL
import core.CellType.DATA
import core.CellType.DELETE
import core.CellType.EDIT
import core.CellType.INFO

@Composable
fun RowScope.TableCell(
    text: String = "",
    weight: Float = 1f,
    type: CellType = DATA,
    onClick: () -> Unit = {},
    modifier: Modifier = Modifier,
) {
    Box(
        modifier = modifier
            .border(1.dp, Color.Black)
            .fillMaxHeight()
            .weight(weight),
        contentAlignment = Alignment.Center,
    ) {
        when (type) {
            DATA -> Text(
                modifier = Modifier
                    .fillMaxHeight()
                    .horizontalScroll(ScrollState(0))
                    .padding(horizontal = 4.dp)
                    .wrapContentHeight(align = Alignment.CenterVertically),
                text = text,
                textAlign = TextAlign.Center,
                softWrap = false,
                fontSize = 18.sp,
            )

            EDIT -> IconButton(onClick) {
                Icon(imageVector = Icons.Default.Edit, "")
            }

            INFO -> IconButton(onClick) {
                Icon(imageVector = Icons.Default.Info, "")
            }

            DELETE -> IconButton(onClick) {
                Icon(imageVector = Icons.Default.Delete, "")
            }

            ACCEPT -> IconButton(onClick) {
                Icon(imageVector = Icons.Default.Check, "")
            }

            CANCEL -> IconButton(onClick) {
                Icon(imageVector = Icons.Default.Close, "")
            }
        }
    }
}