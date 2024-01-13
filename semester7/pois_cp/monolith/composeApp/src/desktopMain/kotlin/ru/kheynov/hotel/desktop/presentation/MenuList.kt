package ru.kheynov.hotel.desktop.presentation

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.material.Button
import androidx.compose.material.ButtonDefaults
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp

@Composable
fun MenuList(
    buttonsNames: List<String>,
    onClick: (Int) -> Unit = {},
    modifier: Modifier = Modifier,
) {
    LazyColumn(
        modifier = modifier.padding(8.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.SpaceAround
    ) {
        itemsIndexed(buttonsNames) { idx, item ->
            Button(
                modifier = Modifier.fillMaxWidth().padding(horizontal = 8.dp),
                border = BorderStroke(1.dp, Color.Green),
                colors = ButtonDefaults.outlinedButtonColors(contentColor = Color.Green),
                onClick = { onClick(idx) },
            ) {
                Text(text = item, color = Color.Black)
            }
        }
    }
}