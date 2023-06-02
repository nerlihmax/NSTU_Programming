package v3.presentation

import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.AlertDialog
import androidx.compose.material.Button
import androidx.compose.material.ExperimentalMaterialApi
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import v3.domain.entities.Teacher

@OptIn(ExperimentalMaterialApi::class)
@Composable
fun DisciplineInfo(
    info: List<Teacher>,
    onDismiss: () -> Unit = {},
    modifier: Modifier = Modifier,
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Преподаватели дисциплины") },
        text = {
            Box(modifier = Modifier.padding(8.dp).verticalScroll(rememberScrollState())) {
                Column {
                    repeat(info.size) {
                        val block = info[it]
                        Column(
                            modifier = Modifier.border(1.dp, Color.Black).padding(vertical = 8.dp),
                            verticalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text(
                                "ФИО: ${block.fullName}", fontSize = 16.sp, color = Color.Black,
                                modifier = Modifier.padding(vertical = 4.dp)
                            )
                            Text(
                                "Отдел: ${block.department.name}", fontSize = 16.sp, color = Color.Black,
                                modifier = Modifier.padding(vertical = 4.dp)
                            )
                            Text(
                                "Должность: ${block.post}", fontSize = 16.sp, color = Color.Black,
                                modifier = Modifier.padding(vertical = 4.dp)
                            )
                            Text(
                                "Дата найма: ${block.hireDate}", fontSize = 16.sp, color = Color.Black,
                                modifier = Modifier.padding(vertical = 4.dp)
                            )
                        }
                    }
                }
            }
        },
        confirmButton = {
            Button(onClick = onDismiss) {
                Text("ОК")
            }
        },
        modifier = modifier,
    )
}