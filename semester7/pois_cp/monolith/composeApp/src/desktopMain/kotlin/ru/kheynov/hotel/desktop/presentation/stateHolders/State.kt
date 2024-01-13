package ru.kheynov.hotel.desktop.presentation.stateHolders

sealed interface State {
    data object Loading : State
    data object Idle : State
    data class Editing(val row: Int) : State
    data object Adding : State
    data class Error(val text: String) : State
}
