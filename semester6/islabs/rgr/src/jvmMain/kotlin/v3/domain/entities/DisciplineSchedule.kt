package v3.domain.entities

import v3.data.entities.Discipline
import v3.data.entities.Teacher

data class DisciplineSchedule(
    val id: Int,
    val discipline: Discipline,
    val teacher: Teacher,
    val hours: Int,
)