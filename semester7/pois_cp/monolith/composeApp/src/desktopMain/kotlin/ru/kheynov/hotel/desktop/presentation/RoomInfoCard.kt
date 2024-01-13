package ru.kheynov.hotel.desktop.presentation

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.Card
import androidx.compose.material.MaterialTheme
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import ru.kheynov.hotel.shared.domain.entities.RoomInfo
import ru.kheynov.hotel.ui.Dimensions

@Composable
fun RoomInfoCard(
    data: RoomInfo,
    onClick: () -> Unit = {},
    modifier: Modifier = Modifier,
) {
    Card(
        modifier = modifier,
        shape = RoundedCornerShape(Dimensions.spaceLarge),
        backgroundColor = MaterialTheme.colors.surface,
        elevation = 4.dp,
    ) {
        Column(
            verticalArrangement = Arrangement.Center,
        ) {
            Row(
                horizontalArrangement = Arrangement.SpaceAround,
                verticalAlignment = androidx.compose.ui.Alignment.CenterVertically,
            ) {
                Text(
                    text = "ID комнаты: ${data.id}"
                )
                Text(
                    text = "Тип номера: ${data.type}"
                )
                Text(
                    text = "Цена: ${data.price}"
                )
            }
        }
    }
}