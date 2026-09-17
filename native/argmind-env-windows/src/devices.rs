//! Which microphones, speakers and cameras this machine has, and which one is the default.
//!
//! **Everything else in this crate takes the default and says nothing about it.** That is the right
//! behaviour for a program that has just started and the wrong one for a person with a headset, a
//! webcam's microphone and a monitor that also claims to be a speaker — argmind would be listening
//! to whichever Windows happened to prefer, and there was no way to see which, let alone change it.
//!
//! ## What a device is here
//!
//! An **id** and a **name**. The id is what opens it and the name is what a person recognises, and
//! they are not the same string: WASAPI ids look like
//! `{0.0.1.00000000}.{9a...}` and mean nothing to anybody. **The id is what gets remembered**,
//! because a friendly name changes when a driver is updated and an endpoint id does not.
//!
//! Cameras are the exception and are remembered by name, because Media Foundation's stable handle
//! is a symbolic link that is even less readable and this project has no measurement saying the
//! name is unstable. Said here rather than discovered later.

use windows::core::PCWSTR;
use windows::Win32::Media::Audio::{eCapture, eConsole, eRender, EDataFlow, IMMDevice,
                                   IMMDeviceEnumerator, MMDeviceEnumerator, DEVICE_STATE_ACTIVE};
use windows::Win32::System::Com::{CoCreateInstance, CLSCTX_ALL, STGM_READ};

/// One thing that can record, play or see.
#[derive(Debug, Clone, PartialEq)]
pub struct Device {
    /// What opens it. Opaque, stable, and never shown to anybody.
    pub id: String,
    /// What a person calls it.
    pub name: String,
    /// Whether this is the one Windows would pick on its own.
    pub default: bool,
}

/// Every active microphone.
pub fn microphones() -> Result<Vec<Device>, String> {
    audio(eCapture)
}

/// Every active speaker.
pub fn speakers() -> Result<Vec<Device>, String> {
    audio(eRender)
}

/// **Active only.** A disabled or unplugged endpoint is one Windows will not open, and offering it
/// in a list is offering a choice that fails after it is made.
fn audio(flow: EDataFlow) -> Result<Vec<Device>, String> {
    unsafe {
        let enumerator: IMMDeviceEnumerator =
            CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)
                .map_err(|e| format!("오디오 장치를 열거하지 못했습니다 ({e})"))?;

        //: The default is asked for separately rather than inferred, because *first in the list*
        //: is not it and never was.
        let preferred = enumerator.GetDefaultAudioEndpoint(flow, eConsole).ok()
            .and_then(|d| id_of(&d));

        let all = enumerator.EnumAudioEndpoints(flow, DEVICE_STATE_ACTIVE)
            .map_err(|e| format!("{e}"))?;
        let how_many = all.GetCount().map_err(|e| format!("{e}"))?;
        let mut out = Vec::new();
        for i in 0..how_many {
            let Ok(device) = all.Item(i) else { continue };
            let Some(id) = id_of(&device) else { continue };
            let name = name_of(&device).unwrap_or_else(|| "이름 없는 장치".into());
            let default = preferred.as_deref() == Some(id.as_str());
            out.push(Device { id, name, default });
        }
        Ok(out)
    }
}

fn id_of(device: &IMMDevice) -> Option<String> {
    unsafe { device.GetId().ok().and_then(|s| s.to_string().ok()) }
}

/// The name a person would recognise, out of the endpoint's property store.
fn name_of(device: &IMMDevice) -> Option<String> {
    unsafe {
        let store = device.OpenPropertyStore(STGM_READ).ok()?;
        let value = store
            .GetValue(&windows::Win32::Devices::FunctionDiscovery::PKEY_Device_FriendlyName)
            .ok()?;
        let text = value.to_string();
        (!text.is_empty()).then_some(text)
    }
}

/// Open one audio endpoint by id, or the default when there is no id or it has gone.
///
/// **Falling back is deliberate and is reported by the `bool`.** A headset that was chosen and then
/// unplugged should not make argmind deaf; it should make argmind use the built-in microphone and
/// be able to say that it did. Silently doing either would be the failure.
pub fn open_audio(flow: EDataFlow, want: Option<&str>) -> Result<(IMMDevice, bool), String> {
    unsafe {
        let enumerator: IMMDeviceEnumerator =
            CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)
                .map_err(|e| format!("{e}"))?;
        if let Some(id) = want.filter(|id| !id.is_empty()) {
            let wide: Vec<u16> = id.encode_utf16().chain(std::iter::once(0)).collect();
            if let Ok(device) = enumerator.GetDevice(PCWSTR(wide.as_ptr())) {
                //: Present in the list is not the same as present at all: `GetDevice` answers for
                //: an endpoint that is unplugged, and opening it fails later and less clearly.
                if device.GetState().map(|s| s == DEVICE_STATE_ACTIVE).unwrap_or(false) {
                    return Ok((device, false));
                }
            }
        }
        let device = enumerator.GetDefaultAudioEndpoint(flow, eConsole)
            .map_err(|e| format!("기본 장치를 열지 못했습니다 ({e})"))?;
        Ok((device, want.is_some_and(|id| !id.is_empty())))
    }
}
