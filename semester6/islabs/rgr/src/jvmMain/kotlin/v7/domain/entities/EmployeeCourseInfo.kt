package v7.domain.entities

import java.time.LocalDate

data class EmployeeCourseInfo(
    val fullName: String,
    val position: String,
    val department: String? = null,
    val course: String,
    val courseDescription: String,
    val duration: Int,
    val startDate: LocalDate,
)
