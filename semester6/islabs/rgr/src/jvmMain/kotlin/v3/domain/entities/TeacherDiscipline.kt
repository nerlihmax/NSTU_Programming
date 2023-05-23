package v3.domain.entities

data class TeacherDiscipline(
    val name: String,
    val discipline: Discipline,
    val group: Group,
    val hours: Int,
)