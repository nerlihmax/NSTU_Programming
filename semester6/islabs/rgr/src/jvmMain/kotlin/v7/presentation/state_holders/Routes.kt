package v7.presentation.state_holders

sealed interface Routes {
    object Departments : Routes
    object Positions : Routes
    object Courses : Routes
    object Employees : Routes
    object CoursesCompletion : Routes
}