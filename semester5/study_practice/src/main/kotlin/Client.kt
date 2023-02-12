import java.net.Socket

fun main() {
    val socket = Socket("localhost", 8080)
    val writer = socket.getOutputStream().writer()
    val reader = socket.getInputStream().reader()
    writer.write("Hello, world!")
    writer.flush()
    println(reader.readText())
}

