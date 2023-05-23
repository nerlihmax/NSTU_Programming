package v3.domain.entities

import v3.data.entities.Department
import java.time.LocalDate

data class Teacher(
    val id: Int,
    val fullName: String,
    val department: Department,
    val post: String,
    val hireDate: LocalDate,
)