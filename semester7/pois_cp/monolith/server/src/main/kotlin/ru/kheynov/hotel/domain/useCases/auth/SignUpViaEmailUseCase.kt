package ru.kheynov.cinemabooking.domain.useCases.auth

import org.koin.core.component.KoinComponent
import org.koin.core.component.inject
import ru.kheynov.cinemabooking.domain.entities.UserDTO
import ru.kheynov.cinemabooking.domain.repositories.UsersRepository
import ru.kheynov.cinemabooking.jwt.hashing.HashingService
import ru.kheynov.cinemabooking.jwt.token.*
import java.util.*

class SignUpViaEmailUseCase : KoinComponent {

    private val usersRepository: UsersRepository by inject()
    private val tokenService: TokenService by inject()
    private val hashingService: HashingService by inject()
    private val tokenConfig: TokenConfig by inject()

    sealed interface Result {
        data class Successful(val tokenPair: TokenPair) : Result
        data object Failed : Result
        data object UserAlreadyExists : Result
    }

    suspend operator fun invoke(user: UserDTO.UserEmailSignUp): Result {
        if (usersRepository.getUserByEmail(user.email) != null) return Result.UserAlreadyExists
        val userId = getRandomUserID()
        val tokenPair = tokenService.generateTokenPair(tokenConfig, TokenClaim("userId", userId))

        val resUser = UserDTO.User(
            userId = userId,
            username = user.username.ifEmpty { "Guest-${getRandomUsername()}" },
            email = user.email,
            passwordHash = hashingService.generateHash(user.password),
            authProvider = "local",
        )
        val registerUserResult = usersRepository.registerUser(resUser)
        val createUserRefreshTokenResult = usersRepository.createRefreshToken(
            userId = userId,
            clientId = user.clientId,
            refreshToken = RefreshToken(
                token = tokenPair.refreshToken.token,
                expiresAt = tokenPair.refreshToken.expiresAt,
            ),
        )

        return if (registerUserResult && createUserRefreshTokenResult) Result.Successful(tokenPair) else Result.Failed
    }
}

fun getRandomUserID(): String = UUID.randomUUID().toString().subSequence(0..7).toString()
fun getRandomUsername(): String = UUID.randomUUID().toString().subSequence(0..6).toString()
