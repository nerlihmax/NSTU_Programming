use crate::{bcd_to_binary, get_group_value, set_group_value};

#[test]
fn test_bcd_to_binary() {
    assert_eq!(bcd_to_binary(0b001001110011), 0b100010001);
    assert_eq!(bcd_to_binary(0b01101000), 0b1000100);
    assert_eq!(bcd_to_binary(0b10010111), 0b1100001);
    assert_eq!(bcd_to_binary(0b01110110), 0b1001100);
    assert_eq!(bcd_to_binary(0b0001100001101001), 0b11101001101);
    assert_eq!(bcd_to_binary(0b10000110010110010110), 0b10101001001000100);
}

#[test]
fn test_get_group_value() {
    assert_eq!(get_group_value(0b0010_0111_0011, 0), 0b0011);
    assert_eq!(get_group_value(0b0010_0111_0011, 1), 0b0111);
    assert_eq!(get_group_value(0b0010_0111_0011, 2), 0b0010);
}

#[test]
fn test_set_group_value() {
    assert_eq!(
        set_group_value(0b0010_0111_0011, 0, 0b0111),
        0b0010_0111_0111
    );
    assert_eq!(
        set_group_value(0b0010_0111_0011, 1, 0b1111),
        0b0010_1111_0011
    );
    assert_eq!(
        set_group_value(0b0010_0111_1001, 0, 0b0110),
        0b0010_0111_0110
    );
}
