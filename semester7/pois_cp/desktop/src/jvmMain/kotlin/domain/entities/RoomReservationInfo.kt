package domain.entities

import data.entities.Room
import data.entities.User
import java.time.LocalDate


data class RoomReservationInfo(
    val id: String,
    val room: Room,
    val user: User,
    val from: LocalDate,
    val to: LocalDate,
)
