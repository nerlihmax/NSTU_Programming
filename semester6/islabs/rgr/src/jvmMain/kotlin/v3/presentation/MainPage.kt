package v3.presentation

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
import androidx.compose.material.CircularProgressIndicator
import androidx.compose.material.ExperimentalMaterialApi
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import core.DataInputDialog
import core.MenuList
import core.Table

@OptIn(ExperimentalMaterialApi::class)
@Composable
fun MainPageV3() {
    val viewModel = remember { MainViewModel() }

    val route = viewModel.route.collectAsState()
    val state = viewModel.state.collectAsState()
    val error = viewModel.errState.collectAsState()
    val data = viewModel.data.collectAsState()

    Row(modifier = Modifier.fillMaxSize()) {
        Column(verticalArrangement = Arrangement.Center, modifier = Modifier.fillMaxHeight()) {
            MenuList(buttonsNames = listOf("Отделы", "Преподаватели", "Группы", "Дисциплины", "Сводка по дисциплинам"),
                modifier = Modifier.width(200.dp).fillMaxHeight(0.6f), onClick = { idx ->
                    when (idx) {
                        0 -> viewModel.navigate(MainViewModel.Route.Departments)
                        1 -> viewModel.navigate(MainViewModel.Route.Teachers)
                        2 -> viewModel.navigate(MainViewModel.Route.Groups)
                        3 -> viewModel.navigate(MainViewModel.Route.Disciplines)
                        4 -> viewModel.navigate(MainViewModel.Route.DisciplinesSchedule)
                        else -> println("ERROR")
                    }
                })
        }
        Column(
            modifier = Modifier.fillMaxHeight().weight(1f),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            if (state.value is MainViewModel.MainState.Loading || data.value.header.isEmpty()) {
                CircularProgressIndicator(modifier = Modifier.size(80.dp))
            } else {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Table(
                        data = data.value, modifier = Modifier.fillMaxHeight(0.6f).padding(horizontal = 24.dp),
                        onEditing = { id ->
                            viewModel.startEditing(id)
                        },
                        onDeleting = { id ->
                            viewModel.deleteRow(id)
                        },
                        onShowInfo = { id ->
                            viewModel.showInfo(id)
                        },
                        infoRequired = route.value is MainViewModel.Route.Teachers || route.value is MainViewModel.Route.Disciplines,
                    )
                    Button(onClick = viewModel::startAdding, modifier = Modifier.padding(8.dp)) {
                        Text("Добавить")
                    }
                }
                if (state.value is MainViewModel.MainState.Editing) {
                    DataInputDialog(
                        header = data.value.header,
                        data = data.value.data.firstOrNull { it.items[0].toInt() == (state.value as MainViewModel.MainState.Editing).row }?.items
                            ?: emptyList(),
                        onEdited = { row ->
                            viewModel.edit(row)
                            viewModel.clearState()
                        },
                        onCanceled = viewModel::clearState,
                    )
                }
                if (state.value is MainViewModel.MainState.TeacherInfo) {
                    val data = (state.value as MainViewModel.MainState.TeacherInfo).data
                    TeacherInfo(
                        info = data,
                        onDismiss = viewModel::hideInfo,
                    )
                }
                if (state.value is MainViewModel.MainState.DisciplineInfo) {
                    DisciplineInfo(
                        info = (state.value as MainViewModel.MainState.DisciplineInfo).data,
                        onDismiss = viewModel::hideInfo,
                    )
                }
                if (state.value is MainViewModel.MainState.Adding) {
                    DataInputDialog(
                        header = data.value.header,
                        onEdited = { row ->
                            viewModel.add(row)
                            viewModel.clearState()
                        },
                        onCanceled = viewModel::clearState,
                    )
                }
                if (error.value is MainViewModel.ErrorState.ShowError) {
                    AlertDialog(
                        onDismissRequest = viewModel::clearState,
                        text = {
                            Text((error.value as? MainViewModel.ErrorState.ShowError)?.error ?: "ERROR")
                        },
                        confirmButton = {
                            Button(onClick = viewModel::clearState) {
                                Text("OK")
                            }
                        },
                    )
                }
            }
        }
    }
}
