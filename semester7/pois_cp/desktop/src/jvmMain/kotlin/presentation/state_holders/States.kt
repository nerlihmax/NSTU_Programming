package presentation.state_holders

import domain.entities.RoomReservationInfo

sealed interface State {
    object Loading : State
    object Idle : State
    data class Editing(val row: String) : State
    object Adding : State
    data class ShowRoomReservations(val data: List<RoomReservationInfo>) : State
    data class ShowRoomBooking(val roomId: String) : State
}
