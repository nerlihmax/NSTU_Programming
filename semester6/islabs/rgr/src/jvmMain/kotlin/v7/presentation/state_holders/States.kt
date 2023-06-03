package v7.presentation.state_holders

import v7.domain.entities.EmployeeCourseInfo

sealed interface State {
    object Loading : State
    object Idle : State
    data class Editing(val row: Int) : State
    object Adding : State
    data class ShowCurrentCourses(val data: List<EmployeeCourseInfo>) : State
    data class ShowPlannedCourses(val data: List<EmployeeCourseInfo>) : State
    data class ShowPassedCourses(val data: List<EmployeeCourseInfo>) : State
}
