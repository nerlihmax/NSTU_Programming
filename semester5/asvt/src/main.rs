#[cfg(test)]
mod test;

pub fn reverse_bits(input: u32) -> u32 { //функция "разворачивания битов" в числе
    let mut output = 0u32; // выходное число

    for i in 0..32 {
        let bit = (input >> i) & 1; //крайний бит после сдвига
        output |= bit << (31 - i); //перемещаем в конец
    }

    output // возврат значения
}

pub fn bcd_to_binary(bcd: u32) -> u32 {
    let mut input = bcd.clone(); // клонируем входное число

    let mut bin = 0u32; // выходное число

    for _ in 0..32 { //цикл по битам
        bin <<= 1; // сдвигаем влево выходное число
        bin += input & 1; // добавляем крайний бит

        input >>= 1; // сдвигаем вправо входное число

        for j in 0..8 { //цикл по группам по 4 бита
            let group = get_group_value(input, j); // получаем значение группы

            if group >= 8 { // если значение группы больше 8
                input = set_group_value(input, j, group - 3); // уменьшаем значение группы на 3
            }
        }
    }

    reverse_bits(bin) // возврат значения (перевернутого)
}

pub fn get_group_value(input: u32, group: u32) -> u32 { // получение значения группы
    let mut group_value = 0u32; // значение группы

    for i in 0..4 { //цикл по битам группы
        let shifted = input.overflowing_shr(group * 4 + i); // сдвигаем входное число на 4 бита вправо

        let bit = shifted.0 & 1; // получаем крайний бит
        group_value |= bit << i; // добавляем крайний бит в значение группы
    }

    group_value // возврат значения
}

pub fn set_group_value(input: u32, group: u32, group_value: u32) -> u32 { // установка значения группы
    let mut output = input; // выходное число
    output = output & !(0b1111 << (group * 4)); // обнуляем группу
    output = output | (group_value << (group * 4)); // устанавливаем значение группы
    output // возврат значения
}

fn main() {
    let mut target = String::new();

    println!("Enter BCD number:");
    std::io::stdin()
        .read_line(&mut target)
        .expect("Error happened");

    if let Some('\n') = target.chars().next_back() {
        target.pop();
    }
    if let Some('\r') = target.chars().next_back() {
        target.pop();
    }

    let target = u32::from_str_radix(target.as_str(), 2).unwrap(); //преобразуем строку в число с основанием 2

    println!("Output: {:b}", bcd_to_binary(target));
}
