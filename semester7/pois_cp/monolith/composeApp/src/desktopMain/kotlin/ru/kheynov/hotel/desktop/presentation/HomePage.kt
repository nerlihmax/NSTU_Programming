package ru.kheynov.hotel.desktop.presentation

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material.AlertDialog
import androidx.compose.material.Button
import androidx.compose.material.ButtonDefaults
import androidx.compose.material.CircularProgressIndicator
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import ru.kheynov.hotel.desktop.presentation.stateHolders.ErrorStates
import ru.kheynov.hotel.desktop.presentation.stateHolders.Routes
import ru.kheynov.hotel.desktop.presentation.stateHolders.State

@Composable
fun HomePage() {
    val viewModel = remember { ViewModel() }

    val route = viewModel.route.collectAsState()
    val state = viewModel.state.collectAsState()
    val error = viewModel.errState.collectAsState()
    val data = viewModel.data.collectAsState()

    Row(modifier = Modifier.fillMaxSize()) {
        Column(verticalArrangement = Arrangement.Center, modifier = Modifier.fillMaxHeight()) {
            MenuList(
                buttonsNames = listOf("Бронирования", "Комнаты"),
                modifier = Modifier.width(200.dp).fillMaxHeight(0.6f), onClick = { idx ->
                    when (idx) {
                        0 -> viewModel.navigate(Routes.Bookings)
                        1 -> viewModel.navigate(Routes.Rooms)
                        else -> println("ERROR")
                    }
                })
        }
        Column(
            modifier = Modifier.fillMaxHeight().weight(1f),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            if (state.value is State.Loading /*|| data.value.header.isEmpty()*/) {
                CircularProgressIndicator(modifier = Modifier.size(80.dp))
            } else {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Box(modifier = Modifier.size(200.dp).background(Color.Red))
                }
            }
            if (error.value is ErrorStates.ShowError) {
                AlertDialog(
                    title = { Text("Ошибка", color = Color.Black) },
                    onDismissRequest = { },
                    text = {
                        Text(
                            text = (error.value as? ErrorStates.ShowError)?.error ?: "ERROR",
                            textAlign = TextAlign.Center, color = Color.Black
                        )
                    },
                    confirmButton = {
                        Button(
                            onClick = viewModel::clearErrorState,
                            border = BorderStroke(1.dp, Color.Green),
                            colors = ButtonDefaults.outlinedButtonColors(contentColor = Color.Green),
                        ) {
                            Text("OK", textAlign = TextAlign.Center, color = Color.Black)
                        }
                    },
                    modifier = Modifier.width(300.dp),
                )
            }
        }
    }
}