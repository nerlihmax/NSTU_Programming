package v7.presentation.state_holders

sealed interface ErrorStates {
    object Idle : ErrorStates
    data class ShowError(val error: String) : ErrorStates
}