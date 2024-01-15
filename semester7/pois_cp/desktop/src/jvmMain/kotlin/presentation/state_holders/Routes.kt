package presentation.state_holders

sealed interface Routes {
    object Hotels : Routes
    object Reservations : Routes
    data class Rooms(
        val hotelId: Int,
    ): Routes
    object Users : Routes
}