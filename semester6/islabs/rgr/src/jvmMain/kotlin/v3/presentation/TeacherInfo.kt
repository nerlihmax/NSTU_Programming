package v3.presentation

import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.AlertDialog
import androidx.compose.material.Button
import androidx.compose.material.ExperimentalMaterialApi
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import v3.domain.entities.TeacherDiscipline

@OptIn(ExperimentalMaterialApi::class)
@Composable
fun TeacherInfo(
    info: List<TeacherDiscipline>,
    onDismiss: () -> Unit = {},
    modifier: Modifier = Modifier,
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Информация о преподавателе") },
        text = {
            LazyColumn(
                modifier = modifier.size(width = 300.dp, height = 500.dp).padding(8.dp),
            ) {
                items(info) {
                    Column(
                        modifier = Modifier
                            .border(1.dp, Color.Black)
                            .padding(vertical = 8.dp)
                    ) {
                        Text(
                            "ФИО: ${it.name}", fontSize = 16.sp, color = Color.Black,
                            modifier = Modifier.padding(vertical = 4.dp)
                        )
                        Text(
                            "Дисциплина: ${it.discipline}", fontSize = 16.sp, color = Color.Black,
                            modifier = Modifier.padding(vertical = 4.dp)
                        )
                        Text(
                            "Группа: ${it.group}", fontSize = 16.sp, color = Color.Black,
                            modifier = Modifier.padding(vertical = 4.dp)
                        )
                        Text(
                            "Часы: ${it.hours}", fontSize = 16.sp, color = Color.Black,
                            modifier = Modifier.padding(vertical = 4.dp)
                        )
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