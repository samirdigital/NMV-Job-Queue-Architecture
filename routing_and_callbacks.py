"""
routing_and_callbacks.py — Extracted from site/app.py

Contains all code that:
  1. Decides which compute tier to use (Modal vs. VPS)
  2. Manages credit accounting that drives the routing decision
  3. Handles the Flask POST/GET callback endpoints that workers report into

These are excerpts — not standalone runnable code. They require the full Flask
app context (db, current_user, modal import, etc.) from site/app.py.
"""

# ---------------------------------------------------------------------------
# Credit tier constants
# ---------------------------------------------------------------------------

timed_credits_info = {
    "free": {'cap': 3, 'reset_by': {'days': 1}},
}


# ---------------------------------------------------------------------------
# FreeCredits DB model
# Tracks daily free-tier credit allotments per user, de-duplicated across
# IP, cookie, and browser fingerprint to prevent gaming the system.
# ---------------------------------------------------------------------------

class FreeCredits(db.Model):
    __tablename__ = "free_credits"
    id = db.Column(db.Integer, primary_key=True)
    main_user_id = db.Column(db.Integer)
    ip = db.Column(db.String)
    cookie = db.Column(db.String)
    fingerprint = db.Column(db.String)
    credits = db.Column(db.Integer, default=3)
    cap = db.Column(db.Integer, default=3)
    next_reset = db.Column(db.Integer, default=0)

    def __init__(self, user, ip=None, cookie=None, fingerprint=None):
        self.main_user_id = user.id
        self.ip = ip
        self.cookie = cookie
        self.fingerprint = fingerprint
        self.next_reset = 0

        if user.subscription == "free":
            self.credits = 3
            self.cap = 3
            self.set_next_reset()

    def set_next_reset(self):
        current_time = time.time()
        while self.next_reset < current_time:
            self.next_reset += 86400  # 24-hour rolling window

    def should_reset_credits(self):
        return time.time() >= self.next_reset

    def reset_credits(self):
        self.credits = self.cap
        self.set_next_reset()
        db.session.commit()

    @staticmethod
    def get_timed_entry(user, ip=None, cookie=None, fingerprint=None, canCreateNew=True):
        """
        Finds the FreeCredits row for this user, matching on any of:
        user_id, cookie, fingerprint, or IP. This cross-linking prevents users
        from creating new accounts to reset their free credits.
        """
        associated = {
            'user_id': FreeCredits.query.filter_by(main_user_id=user.id).first(),
            'cookie': FreeCredits.query.filter_by(cookie=cookie).first() if cookie else None,
            'fingerprint': FreeCredits.query.filter_by(fingerprint=fingerprint).first() if fingerprint else None,
            'ip': FreeCredits.query.filter_by(ip=ip).first() if ip else None,
        }

        matched_entry = None
        for key, value in associated.items():
            if value is not None:
                matched_entry = value
                if matched_entry.main_user_id != user.id:
                    matched_entry.main_user_id = user.id
                    db.session.commit()
                break

        if matched_entry is not None:
            if matched_entry.should_reset_credits():
                matched_entry.reset_credits()
            return matched_entry
        else:
            if not canCreateNew:
                return 303
            new_entry = FreeCredits(user, ip, cookie, fingerprint)
            db.session.add(new_entry)
            db.session.commit()
            return new_entry

    @classmethod
    def get_timed_credits(cls, user, ip=None, cookie=None, fingerprint=None) -> int:
        return FreeCredits.get_timed_entry(user, ip, cookie, fingerprint).credits


# ---------------------------------------------------------------------------
# ConversionCharge — result object from User.charge()
# Carries the tier label ("paid" vs "free") which drives routing downstream.
# ---------------------------------------------------------------------------

class ConversionCharge():
    def __init__(self, user, conversion_type=None, conversion_amount=None, success=None):
        self.user = user
        self.success = success
        self.conversion_type = conversion_type   # "paid" or "free"
        self.conversion_amount = conversion_amount


# ---------------------------------------------------------------------------
# Conversion DB model
# One row per inference job. modal_entity stores either:
#   - A Modal FunctionCall object_id  (paid tier, or free-tier Modal fallback)
#   - The string "httprequest"         (free tier, successfully sent to VPS)
# ---------------------------------------------------------------------------

class Conversion(db.Model):
    __tablename__ = "conversions"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(length=255))
    model_name = db.Column(db.String(length=255))
    duration = db.Column(db.Integer, default=0)
    timestamp = db.Column(db.Integer, default=int(time.time()))
    status = db.Column(db.String())          # JSON-encoded status from worker callbacks
    uuid = db.Column(db.String(length=255))  # Public-facing unique ID; prefixed "free-" for free tier
    downloads = db.Column(db.String())       # CDN URLs of final output files
    modal_entity = db.Column(db.String(), default=None)
    ip = db.Column(db.String(), default=None)


# ---------------------------------------------------------------------------
# User.charge() — credit deduction with tier determination
#
# Credit priority order:
#   1. Permanent paid credits (_credits)
#   2. Expiring paid credits (expiring_credits)
#   3. Daily free credits (FreeCredits table)
#
# The returned ConversionCharge.conversion_type ("paid" or "free") is what
# the routing logic reads to decide Modal vs. VPS.
# ---------------------------------------------------------------------------

# (Method on the User model — shown here as a standalone function for clarity)
def charge(self, amount: int, real=True) -> ConversionCharge:
    if isinstance(self.credits, tuple) or isinstance(amount, tuple):
        print("ERROR: Either self.credits or amount is a tuple.")
        return ConversionCharge(self, success=False)

    if self.credits < amount:
        return ConversionCharge(self, success=False)
    elif self.credits == inf:
        # Admin / unlimited accounts — always paid routing, never deducted
        return ConversionCharge(self, success=True, conversion_type="paid", conversion_amount=amount)
    elif self._credits >= amount:
        self._credits -= (amount if real else 0)
        db.session.commit()
        return ConversionCharge(self, success=True, conversion_type="paid", conversion_amount=amount)
    elif self.expiring_credits >= amount:
        self.expiring_credits -= (amount if real else 0)
        db.session.commit()
        return ConversionCharge(self, success=True, conversion_type="paid", conversion_amount=amount)
    elif FreeCredits.get_timed_entry(self).credits >= amount:
        # Charge from daily free allotment → routes to VPS queue
        FreeCredits.get_timed_entry(self).credits -= (amount if real else 0)
        return ConversionCharge(self, success=True, conversion_type="free", conversion_amount=amount)
    else:
        return ConversionCharge(self, success=False)


# ---------------------------------------------------------------------------
# Two-tier routing logic  (inside the modelPagePost handler in site/app.py)
#
# How it works:
#   - charge.conversion_type == "free"  →  UUID gets "free-" prefix
#   - "free" in unique                  →  send to VPS endpoint
#   - VPS unreachable                   →  fall back to Modal
#   - paid tier                         →  send directly to Modal
#
# updateLink is the Flask callback URL that workers POST status into.
# The IP substitution replaces internal/localhost addresses with the
# public IP so Modal (running in the cloud) can reach the Flask server.
# ---------------------------------------------------------------------------

# unique = str(uuid.uuid4()).split("-")[0]

if charge.conversion_type == "free":
    unique = "free-" + unique

conversion = Conversion(
    model_name=model.artist,
    user_id=current_user.id,
    timestamp=int(time.time()),
    status="{}",
    uuid=unique,
    ip=request.remote_addr
)
db.session.add(conversion)
db.session.commit()

conversion_id = conversion.id
updateLink = (
    url_for("conversionUpdatePost", id=conversion_id, _external=True)
    .replace("localhost", os.environ['PUBLIC_IP'])
    .replace("10.0.0.84", os.environ['PUBLIC_IP'])
)

modal_entity = None
with open(tempLocation, "rb") as f:
    if "free" not in unique:
        # PAID TIER: spawn on Modal serverless GPU
        modal_entity = modal.Function.lookup('rvc', "interface").spawn(
            model.artist, f.read(),
            transpose=transpose,
            extract=extract,
            updateLink=updateLink
        )
        conversion.modal_entity = modal_entity.object_id
    else:
        # FREE TIER: try self-hosted VPS first (cheaper), fall back to Modal
        try:
            req = requests.post(
                "https://california.notmyvoice.ai/interface",
                params={
                    "model_name": model.artist,
                    "transpose": transpose,
                    "extract": extract,
                    "update_link": updateLink,
                },
                files={"audio": f}
            )
            conversion.modal_entity = "httprequest"
        except Exception as e:
            print("COULDNT USE FREE SERVER: {}".format(e))
            # VPS unreachable — fall back to Modal (user still gets service)
            modal_entity = modal.Function.lookup('rvc', "interface").spawn(
                model.artist, f.read(),
                transpose=transpose,
                extract=extract,
                updateLink=updateLink
            )
            conversion.modal_entity = modal_entity.object_id

db.session.commit()

# Response tells the client where to poll for updates
return json.jsonify({
    "getUpdates": url_for('conversionGetUpdates', id=conversion_id, nocache="nocache", _external=True),
    'id': conversion_id,
    'modal_entity': conversion.modal_entity,
    'uuid': unique,
    'publicLink': url_for('conversionPage', unique=unique, _external=True)
})


# ---------------------------------------------------------------------------
# Flask callback endpoints
# Workers (both Modal and VPS) POST status updates here as jobs progress.
# The client polls GET /conversionUpdate/<id> to display live progress.
# ---------------------------------------------------------------------------

@app.route("/conversionUpdate/<id>", methods=["POST"])
def conversionUpdatePost(id):
    """
    Receives status updates from workers (Modal or VPS).
    Payload: {"update": {"status": "...", "description": "..."}}
    When status == "Done", description contains the CDN download URLs.
    """
    conversion = Conversion.query.filter_by(id=id).first()
    if not conversion:
        return "Conversion not found.", 404
    if "done" in conversion.status.lower():
        return "Conversion already done.", 200
    import json
    conversion.status = json.dumps(request.json.get('update'))
    if request.json.get('update').get('status') == 'Done':
        conversion.downloads = request.json.get('update').get('description')
    print(conversion.status)
    db.session.commit()
    return "Success", 200


@app.route("/conversionUpdate/<id>", methods=["GET"])
@app.route("/conversionUpdate/<id>/<nocache>", methods=["GET"])
def conversionGetUpdates(id, nocache=None):
    """
    Client polling endpoint. Returns current job status as JSON.

    For paid-tier jobs (modal_entity is a FunctionCall ID): also probes Modal
    directly to detect silent ERROR returns that bypass the callback.

    On error: automatically refunds 1 credit to the user.
    """
    conversion = Conversion.query.filter_by(id=int(id)).first()
    if conversion.status.upper() == "ERROR":
        return {"status": "❌ Error", "description": "Something went wrong. You've been refunded. Try again!", "link": "mailto:hello@notmyvoice.ai"}, 200

    try:
        # For paid-tier Modal jobs: check the FunctionCall directly
        modal_entity = modal.functions.FunctionCall.from_id(conversion.modal_entity)
        result = modal_entity.get(1)
    except:
        result = ""

    print(result)
    if result == "ERROR" or conversion.status.upper() == "ERROR":
        conversion.status = "ERROR"
        user = User.query.filter_by(id=conversion.user_id).first()
        if user is None:
            user = User.query.filter_by(discord_id=conversion.user_id).first()
            if user is None:
                print("?????")
        user._credits += 1   # Refund on error
        db.session.commit()
        return {
            "status": "❌ Error",
            "description": "Something went wrong. You've been refunded. Try again!",
            "link": "mailto:hello@notmyvoice.ai",
        }, 200

    wow = json.loads(conversion.status)
    wow['link'] = url_for('conversionPage', unique=conversion.uuid, _external=True)
    db.session.commit()
    return wow, 200
