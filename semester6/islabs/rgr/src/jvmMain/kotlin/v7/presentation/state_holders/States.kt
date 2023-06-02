package v7.presentation.state_holders

sealed interface State {
    object Loading : State
    object Idle : State
    data class Editing(val row: Int) : State
    object Adding : State
//    data class Info1(val data: List<TeacherDiscipline>) : State
//    data class Info2(val data: List<Teacher>) : State
}
