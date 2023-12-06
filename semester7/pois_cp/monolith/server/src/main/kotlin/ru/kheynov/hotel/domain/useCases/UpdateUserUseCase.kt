package ru.kheynov.hotel.domain.useCases

import org.koin.core.component.KoinComponent
import org.koin.core.component.inject
import ru.kheynov.hotel.domain.entities.UserUpdate
import ru.kheynov.hotel.domain.repository.UsersRepository

class UpdateUserUseCase : KoinComponent {
    private val usersRepository: UsersRepository by inject()

    sealed interface Result {
        data object Successful : Result
        data object Failed : Result
        data object UserNotExists : Result
        data object AvatarNotFound : Result
    }

    suspend operator fun invoke(
        userId: String,
        update: UserUpdate,
    ): Result {
        if (usersRepository.getUserByID(userId) == null) return Result.UserNotExists
        return if (usersRepository.updateUserByID(
                userId,
                update,
            )
        ) {
            Result.Successful
        } else {
            Result.Failed
        }
    }
}