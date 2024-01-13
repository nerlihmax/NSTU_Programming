package ru.kheynov.hotel.desktop.presentation.stateHolders

sealed interface ErrorStates {
    data object Idle : ErrorStates
    data class ShowError(val error: String) : ErrorStates
}