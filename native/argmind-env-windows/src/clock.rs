//! **What the date is here**, which is a question about the machine and not about UTC.
//!
//! An errand is told today's date because the first one asked to look up the weather guessed it —
//! *2023년 10월 5일* — and the date a person means is the one on their wall.

use windows::Win32::System::Time::{GetTimeZoneInformation, TIME_ZONE_INFORMATION};

/// Minutes to add to UTC to get the time on this machine's clock, **daylight saving included**.
/// `0` if Windows will not say, which gives the UTC date — at worst a day off near midnight.
pub fn utc_offset_minutes() -> i64 {
    let mut zone = TIME_ZONE_INFORMATION::default();
    //: 0 standard-only, 1 standard now, 2 daylight now, `u32::MAX` failed. Windows' bias is UTC
    //: minus local, so the offset is its negation.
    let which = unsafe { GetTimeZoneInformation(&mut zone) };
    let extra = match which {
        1 => zone.StandardBias,
        2 => zone.DaylightBias,
        0 => 0,
        _ => return 0,
    };
    -((zone.Bias + extra) as i64)
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_offset_is_a_real_time_zone() {
        //: Every zone on Earth is between UTC-12 and UTC+14.
        let m = super::utc_offset_minutes();
        assert!((-12 * 60..=14 * 60).contains(&m), "{m}");
    }
}
