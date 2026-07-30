"""Live-arm fixtures, C2PA side (runs in the bench venv: c2pa-python 0.37.2 + cryptography).

Reordered per the 0.0.3 runway: this arm no longer waits behind the watermark one. Builds a
real signed fixture (local CA chain, digicert TSA) asserting trainedAlgorithmicMedia, plus
two F1 probes: what the SDK ACTUALLY raises on (a) a manifest-less file and (b) a signed
file with its manifest bytes corrupted; the narrowed _reads_as_no_manifest set is designed
against those measured messages, not guessed."""

import datetime as dt
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

FIX = Path(__file__).parent / "fixtures"
now = dt.datetime.now(dt.timezone.utc)

ca_key = ec.generate_private_key(ec.SECP256R1())
ca_name = x509.Name([
    x509.NameAttribute(NameOID.COMMON_NAME, "gaige local test root"),
    x509.NameAttribute(NameOID.ORGANIZATION_NAME, "gaige-test"),
    x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
])
ca_cert = (
    x509.CertificateBuilder()
    .subject_name(ca_name).issuer_name(ca_name)
    .public_key(ca_key.public_key())
    .serial_number(x509.random_serial_number())
    .not_valid_before(now - dt.timedelta(days=1))
    .not_valid_after(now + dt.timedelta(days=3650))
    .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
    .add_extension(
        x509.KeyUsage(digital_signature=True, content_commitment=False, key_encipherment=False,
                      data_encipherment=False, key_agreement=False, key_cert_sign=True,
                      crl_sign=True, encipher_only=False, decipher_only=False),
        critical=True)
    .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
    .sign(ca_key, hashes.SHA256())
)
key = ec.generate_private_key(ec.SECP256R1())
name = x509.Name([
    x509.NameAttribute(NameOID.COMMON_NAME, "gaige local test signer"),
    x509.NameAttribute(NameOID.ORGANIZATION_NAME, "gaige-test"),
    x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
])
cert = (
    x509.CertificateBuilder()
    .subject_name(name).issuer_name(ca_name)
    .public_key(key.public_key())
    .serial_number(x509.random_serial_number())
    .not_valid_before(now - dt.timedelta(days=1))
    .not_valid_after(now + dt.timedelta(days=365))
    .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
    .add_extension(
        x509.KeyUsage(digital_signature=True, content_commitment=False, key_encipherment=False,
                      data_encipherment=False, key_agreement=False, key_cert_sign=False,
                      crl_sign=False, encipher_only=False, decipher_only=False),
        critical=True)
    .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.EMAIL_PROTECTION]), critical=True)
    .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
    .add_extension(
        x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
    .sign(ca_key, hashes.SHA256())
)
cert_pem = cert.public_bytes(serialization.Encoding.PEM) + ca_cert.public_bytes(serialization.Encoding.PEM)
key_pem = key.private_bytes(serialization.Encoding.PEM,
                            serialization.PrivateFormat.PKCS8,
                            serialization.NoEncryption())

from c2pa import Builder, C2paSignerInfo, Signer  # noqa: E402

manifest = {
    "claim_generator_info": [{"name": "gaige-fixture-generator", "version": "0.0.1"}],
    "title": "AI-generated test asset",
    "assertions": [
        {"label": "c2pa.actions",
         "data": {"actions": [{
             "action": "c2pa.created",
             "digitalSourceType":
                 "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
             "softwareAgent": "gaige-fixture-generator 0.0.1"}]}},
    ],
}

signed = FIX / "c2pa_signed.jpg"
info = C2paSignerInfo(alg=b"es256", sign_cert=cert_pem, private_key=key_pem,
                      ta_url=b"http://timestamp.digicert.com")
signer = Signer.from_info(info)
builder = Builder(manifest)
if signed.exists():
    signed.unlink()
builder.sign_file(str(FIX / "plain.jpg"), str(signed), signer)
print(f"[fixture] {signed.name} ({signed.stat().st_size} bytes)")

# ---- F1 probes: measure what the SDK actually raises ----
from c2pa import Reader  # noqa: E402

print("\n[f1-probe] manifest-less png:")
try:
    Reader(str(FIX / "plain.png"))
    print("  no exception (reader opened)")
except Exception as e:
    print(f"  {type(e).__name__}: {e!r}")

print("[f1-probe] manifest-less jpg:")
try:
    Reader(str(FIX / "plain.jpg"))
    print("  no exception (reader opened)")
except Exception as e:
    print(f"  {type(e).__name__}: {e!r}")

print("[f1-probe] signed jpg, manifest bytes corrupted:")
data = bytearray(signed.read_bytes())
# find the jumbf region and flip bytes well inside it (past the header magic)
idx = data.find(b"c2pa")
for off in range(200, 220):
    data[idx + off] ^= 0xFF
corrupt = FIX / "c2pa_corrupt.jpg"
corrupt.write_bytes(bytes(data))
try:
    r = Reader(str(corrupt))
    print(f"  no exception; validation_state={r.get_validation_state()!r}")
except Exception as e:
    print(f"  {type(e).__name__}: {e!r}")

print("[f1-probe] non-media file (.txt):")
t = FIX / "note.txt"
t.write_text("plain text file", encoding="utf-8")
try:
    Reader(str(t))
    print("  no exception (reader opened)")
except Exception as e:
    print(f"  {type(e).__name__}: {e!r}")
