# Official Bonn EEG Dataset Download Report

**Date Generated:** 2026-10-10 04:36:27 UTC  
**Official Download Portal:** [https://www.ukbonn.de/epileptologie/arbeitsgruppen/ag-lehnertz-neurophysik/downloads/](https://www.ukbonn.de/epileptologie/arbeitsgruppen/ag-lehnertz-neurophysik/downloads/)  
**Authoritative Source Reference:** Andrzejak RG, Lehnertz K, Rieke C, Mormann F, David P, Elger CE (2001) *Indications of nonlinear deterministic and finite dimensional structures in time series of brain electrical activity: Dependence on recording region and brain state*, Phys. Rev. E 64, 061907.  

## 1. Archive Download Summary

| Subset | Archive Name | HTTP Status | Content-Type | Size (Bytes) | SHA-256 Hash | Integrity & Status |
|:---:|:---:|:---:|:---:|---:|:---:|:---:|
| **A** | `z.zip` | None | `None` | 591,469 | `dd7dd53b84fdbdfde16915c608b8333e926112944861aab933f1b76cedb8ad05` | ✅ Verified OK |
| **B** | `o.zip` | None | `None` | 642,112 | `82b806f39aa04bf010640154dc1bbd548fc8defc2f267653070aaf09129e72da` | ✅ Verified OK |
| **C** | `n.zip` | None | `None` | 587,464 | `98ab85976b5cc8e0b77cc4d4f76a98c6457ea58184751f6f4141a8c30ff5a2c1` | ✅ Verified OK |
| **D** | `f.zip` | None | `None` | 596,691 | `f6cc5a5ce9bcb1f16a00cbfde7bb501cd7fe3874b549118182501aa3d509d578` | ✅ Verified OK |
| **E** | `s.zip` | None | `None` | 787,492 | `befda5a0f352e3bd222b2a8dafa715eb4dc6337ac2bf7a050de62a467615fc3a` | ✅ Verified OK |

## 2. Subset Source Details and Clinical Context

### Subset A (`z.zip`)
- **Direct Official URL:** `https://www.ukbonn.de/site/assets/files/21874/z.zip`
- **Clinical Context (Official Docs):** Healthy volunteers, eyes open, surface EEG
- **Archive Size:** 591,469 bytes (577.61 KB)
- **Archive SHA-256:** `dd7dd53b84fdbdfde16915c608b8333e926112944861aab933f1b76cedb8ad05`
- **Download Timestamp:** `2026-10-10T04:31:42.279254+00:00`
- **Integrity Verification:** Tested using `zipfile.ZipFile.testzip()` (no bad CRC / corrupted members). Magic bytes confirmed `PK\x03\x04`.
- **Error / Retry Log:** None (retries: 0)

### Subset B (`o.zip`)
- **Direct Official URL:** `https://www.ukbonn.de/site/assets/files/21872/o.zip`
- **Clinical Context (Official Docs):** Healthy volunteers, eyes closed, surface EEG
- **Archive Size:** 642,112 bytes (627.06 KB)
- **Archive SHA-256:** `82b806f39aa04bf010640154dc1bbd548fc8defc2f267653070aaf09129e72da`
- **Download Timestamp:** `2026-10-10T04:31:47.976002+00:00`
- **Integrity Verification:** Tested using `zipfile.ZipFile.testzip()` (no bad CRC / corrupted members). Magic bytes confirmed `PK\x03\x04`.
- **Error / Retry Log:** None (retries: 0)

### Subset C (`n.zip`)
- **Direct Official URL:** `https://www.ukbonn.de/site/assets/files/21871/n.zip`
- **Clinical Context (Official Docs):** Epileptic patients, interictal, intracranial EEG (opposite hemisphere / hippocampal formation)
- **Archive Size:** 587,464 bytes (573.70 KB)
- **Archive SHA-256:** `98ab85976b5cc8e0b77cc4d4f76a98c6457ea58184751f6f4141a8c30ff5a2c1`
- **Download Timestamp:** `2026-10-10T04:31:53.004956+00:00`
- **Integrity Verification:** Tested using `zipfile.ZipFile.testzip()` (no bad CRC / corrupted members). Magic bytes confirmed `PK\x03\x04`.
- **Error / Retry Log:** None (retries: 0)

### Subset D (`f.zip`)
- **Direct Official URL:** `https://www.ukbonn.de/site/assets/files/21870/f.zip`
- **Clinical Context (Official Docs):** Epileptic patients, interictal, intracranial EEG (within epileptogenic zone)
- **Archive Size:** 596,691 bytes (582.71 KB)
- **Archive SHA-256:** `f6cc5a5ce9bcb1f16a00cbfde7bb501cd7fe3874b549118182501aa3d509d578`
- **Download Timestamp:** `2026-10-10T04:32:08.940312+00:00`
- **Integrity Verification:** Tested using `zipfile.ZipFile.testzip()` (no bad CRC / corrupted members). Magic bytes confirmed `PK\x03\x04`.
- **Error / Retry Log:** None (retries: 0)

### Subset E (`s.zip`)
- **Direct Official URL:** `https://www.ukbonn.de/site/assets/files/21875/s.zip`
- **Clinical Context (Official Docs):** Epileptic patients, ictal activity, intracranial EEG
- **Archive Size:** 787,492 bytes (769.04 KB)
- **Archive SHA-256:** `befda5a0f352e3bd222b2a8dafa715eb4dc6337ac2bf7a050de62a467615fc3a`
- **Download Timestamp:** `2026-10-10T04:32:12.512879+00:00`
- **Integrity Verification:** Tested using `zipfile.ZipFile.testzip()` (no bad CRC / corrupted members). Magic bytes confirmed `PK\x03\x04`.
- **Error / Retry Log:** None (retries: 0)

## 3. Protocol and Safety Compliance
- **Streaming Downloads:** HTTP streaming chunks (32 KB) were used to guarantee zero risk of RAM exhaustion.
- **Atomic Renaming:** Downloads were buffered to `.part` temporary files before atomic promotion upon SHA-256 and ZIP validation.
- **Content-Type Enforcement:** Verified `application/zip` MIME type to strictly prevent saving HTML error pages.
- **Pristine Preservation:** Original archives are stored unmodified in `data/bonn_eeg/archives/`.

