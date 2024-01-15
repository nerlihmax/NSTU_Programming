package presentation

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material.AlertDialog
import androidx.compose.material.Button
import androidx.compose.material.ButtonDefaults
import androidx.compose.material.CircularProgressIndicator
import androidx.compose.material.ExperimentalMaterialApi
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import core.DataInputDialog
import core.DropDownSelection
import core.MenuList
import core.Table
import presentation.state_holders.ErrorStates
import presentation.state_holders.Routes
import presentation.state_holders.State

@OptIn(ExperimentalMaterialApi::class)
@Composable
fun MainPage() {
    val viewModel = remember { ViewModel() }

    val route = viewModel.route.collectAsState()
    val state = viewModel.state.collectAsState()
    val error = viewModel.errState.collectAsState()
    val data = viewModel.data.collectAsState()
    val hotels = viewModel.hotels.collectAsState()

    Row(modifier = Modifier.fillMaxSize()) {
        Column(verticalArrangement = Arrangement.Center, modifier = Modifier.fillMaxHeight()) {
            MenuList(buttonsNames = listOf("Отели", "Комнаты", "Бронирования", "Гости"),
                modifier = Modifier.width(200.dp).fillMaxHeight(0.6f),
                onClick = { idx ->
                    when (idx) {
                        0 -> viewModel.navigate(Routes.Hotels)
                        1 -> viewModel.navigate(Routes.Rooms(1))
                        2 -> viewModel.navigate(Routes.Reservations)
                        3 -> viewModel.navigate(Routes.Users)
                        else -> println("ERROR")
                    }
                })
        }
        Column(
            modifier = Modifier.fillMaxHeight().weight(1f),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            if (state.value is State.Loading || data.value.header.isEmpty()) {
                CircularProgressIndicator(modifier = Modifier.size(80.dp))
            } else {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    if (route.value is Routes.Rooms) {
                        DropDownSelection(
                            header = hotels.value.find { it.id == (route.value as Routes.Rooms).hotelId }?.name ?: "Выберите отель",
                            data = hotels.value.map { it.name },
                            onSelection = { idx ->
                                viewModel.navigate(Routes.Rooms(hotels.value.find { it.name == idx }?.id ?: 1))
                            },
                            modifier = Modifier.width(200.dp).padding(8.dp),
                        )
                    }

                    Table(
                        data = data.value, modifier = Modifier.fillMaxHeight(0.9f).padding(horizontal = 24.dp),
                        onEditing = { id ->
                            viewModel.startEditing(id)
                        },
                        onDeleting = { id ->
                            viewModel.deleteRow(id)
                        },
                        bookingEnabled = route.value is Routes.Rooms,
                        onBook = { id ->
                            viewModel.showRoomBooking(id)
                        },
                    )
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween,
                    ) {
                        if(route.value != Routes.Reservations){
                            Button(
                                border = BorderStroke(1.dp, Color.Green),
                                colors = ButtonDefaults.outlinedButtonColors(contentColor = Color.Green),
                                onClick = viewModel::startAdding,
                                modifier = Modifier.padding(8.dp),
                            ) {
                                Text("ADD", color = Color.Black)
                            }
                        }

//                        if (route.value is Routes.Employees || route.value is Routes.CoursesCompletion){
//                            Button(
//                                border = BorderStroke(1.dp, Color.Green),
//                                colors = ButtonDefaults.outlinedButtonColors(contentColor = Color.Green),
//                                onClick = viewModel::showCurrentCourses,
//                                modifier = Modifier.padding(8.dp),
//                            ) {
//                                Text("Текущие курсы", color = Color.Black)
//                            }
//                            Button(
//                                border = BorderStroke(1.dp, Color.Green),
//                                colors = ButtonDefaults.outlinedButtonColors(contentColor = Color.Green),
//                                onClick = viewModel::showPlannedCourses,
//                                modifier = Modifier.padding(8.dp),
//                            ) {
//                                Text("Запланированные курсы", color = Color.Black)
//                            }
//                        }
                    }
                }

                when (state.value) {
                    State.Adding -> if(route.value != Routes.Reservations) DataInputDialog(
                        header = data.value.header,
                        onEdited = { row ->
                            viewModel.add(row)
                            viewModel.clearState()
                        },
                        editDenied = run {
                            when(route.value){
                                Routes.Reservations -> listOf(0, 3)
                                is Routes.Rooms -> listOf(0, 4)
                                else -> listOf(0)
                            }
                        },
                        onCanceled = viewModel::clearState,
                    )

                    is State.Editing -> DataInputDialog(
                        header = data.value.header,
                        data = data.value.data.firstOrNull { it.items[0] == (state.value as State.Editing).row }?.items
                            ?: emptyList(),
                        onEdited = { row ->
                            viewModel.edit(row)
                            viewModel.clearState()
                        },
                        editDenied = run {
                            when(route.value){
                                Routes.Reservations -> listOf(0, 3)
                                is Routes.Rooms -> listOf(0, 4)
                                else -> listOf(0)
                            }
                        },
                        onCanceled = viewModel::clearState,
                    )

//                    is State.ShowCurrentCourses ->
//                        ReservationsInfo(
//                            info = (state.value as State.ShowCurrentCourses).data,
//                            onDismiss = viewModel::hideInfo,
//                        )
//
//                    is State.ShowPassedCourses ->
//                        ReservationsInfo(
//                            info = (state.value as State.ShowPassedCourses).data,
//                            onDismiss = viewModel::hideInfo,
//                        )
//
//                    is State.ShowPlannedCourses ->
//                        ReservationsInfo(
//                            info = (state.value as State.ShowPlannedCourses).data,
//                            onDismiss = viewModel::hideInfo,
//                        )

                    else -> {}
                }

                if (error.value is ErrorStates.ShowError) {
                    AlertDialog(
                        title = { Text("Ошибка", color = Color.Black) },
                        onDismissRequest = { },
                        text = {
                            Text(
                                text = (error.value as? ErrorStates.ShowError)?.error ?: "ERROR",
                                textAlign = TextAlign.Center,
                                color = Color.Black
                            )
                        },
                        confirmButton = {
                            Button(
                                onClick = viewModel::clearState,
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
}