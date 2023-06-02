package v3.domain.entities

import core.DataRow
import core.TableData
import v3.data.entities.Department
import java.time.LocalDate

data class Teacher(
    val id: Int,
    val fullName: String,
    val department: Department,
    val post: String,
    val hireDate: LocalDate,
)

val List<Teacher>.asTableData: TableData
    get() = TableData(
        header = listOf("ID", "Имя", "Отдел", "Должность", "Дата найма"),
        data = map {
            DataRow(
                listOf(
                    it.id.toString(),
                    it.fullName,
                    it.department.name,
                    it.post,
                    it.hireDate.toString()
                )
            )
        }
    )