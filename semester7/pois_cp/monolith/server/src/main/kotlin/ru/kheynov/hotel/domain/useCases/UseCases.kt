package ru.kheynov.hotel.domain.useCases

import ru.kheynov.hotel.domain.useCases.auth.LoginViaEmailUseCase
import ru.kheynov.hotel.domain.useCases.auth.RefreshTokenUseCase
import ru.kheynov.hotel.domain.useCases.auth.SignUpViaEmailUseCase

class UseCases {
    val signUpViaEmailUseCase = SignUpViaEmailUseCase()
    val loginViaEmailUseCase = LoginViaEmailUseCase()
    val refreshTokenUseCase = RefreshTokenUseCase()
    val deleteUserUseCase = DeleteUserUseCase()
    val updateUserUseCase = UpdateUserUseCase()
    val getUserDetailsUseCase = GetUserDetailsUseCase()
}