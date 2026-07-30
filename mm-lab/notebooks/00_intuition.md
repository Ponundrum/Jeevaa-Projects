# 00 · Intuition — the interview answers

The whole project in plain language, no equations beyond the two Avellaneda–Stoikov
formulas. If any answer here isn't comfortable to say out loud, that part of the project
isn't finished.

---

**1. Why does a market maker who quotes symmetrically around the mid lose money, even
though they buy low and sell high on every round trip?**

Because their fills aren't random — they're *selected against*. You get lifted on your ask
precisely when buyers know something and the price is about to rise, and hit on your bid
right before it falls. So the "buy low, sell high" is an illusion: you bought just before
lower, and sold just before higher. With real BTCUSDT-futures quotes, the numbers are stark:
a maker resting at the touch captures a half-spread of ~0.014 bps but is marked out ~0.30 bps
within seconds — the market takes back roughly *twenty times* what you earned on the fill. So
quoting at the touch genuinely loses. The escape is not to quote symmetrically-but-wider blindly;
it is to *skew* around inventory (the next answers) and to quote wide enough that the captured
spread beats the adverse selection — a width the fill elasticity `κ` pins down.

**2. What is the reservation price, and why isn't it the mid?**

It's the price at which *you* are indifferent to holding your current inventory —
`r = S − q·γσ²(T−t)`. It's the mid only when you're flat. When you're long it sits *below*
the mid (you'd accept a bit less to sell and get flat); when short, above. You quote your
bid and ask symmetrically around `r`, not around the mid, so your whole quote skews toward
unloading inventory. It's the mechanism that turns "manage inventory" into an actual number.

**3. Why does the optimal spread have two terms? What does each one do?**

`δ* = γσ²(T−t) + (2/γ)·ln(1+γ/κ)`.
- The first is the **inventory-risk premium**: extra width you demand because a fill hands
  you a position you'll carry through volatile time. It grows with vol, risk aversion, and
  time left to be wrong, and vanishes at the horizon.
- The second is the **fill/edge tradeoff**: the width you'd want *even with zero inventory
  risk*, because filling at the mid earns nothing and `κ` tells you how fast quoting wider
  costs you fills. It's constant in time.

**4. What happens to the AS quotes as `γ → 0`? As `T − t → 0`? As `σ` doubles?**

- `γ → 0` (risk-neutral): the inventory skew vanishes (reservation → mid) and the spread
  collapses to `2/κ` — pure fill economics, no inventory caution.
- `T − t → 0`: the inventory-risk term → 0, so near the horizon you stop demanding a premium
  for holding (there's no time left to be hurt). In a 24/7 market this is awkward, which is
  why this project freezes `T − t` at a fixed risk horizon.
- `σ` doubles: the inventory-risk term quadruples (`σ²`) — you widen sharply and skew harder,
  because inventory is now four times as dangerous.

**5. What is adverse selection, and how would you measure it with data you have?**

It's the tendency of the market to move against whoever provided liquidity, because the
counterparty was better informed. You measure it with **markouts**: for every trade, look at
where the mid is 1s / 5s / … / 300s later, signed so that "market moved against the passive
side" is negative. Average across trades, in bps. Against the real `bookTicker` mid the curve
is negative from the first second (~−0.37 bps at 1s) and then **flat** at ~−0.30 bps out to
300s — adverse selection is essentially fully realized within ~5 seconds and does not keep
accumulating. The mid you measure against matters: a lagged 1-second trade-price *proxy* put
the 1s point at *+0.6 bps* (a sign-flipped artefact) even though it got the 5–300s plateau
right — which is why real quotes are worth the extra 188 MB/day.

**6. Your inventory is +10 and volatility just doubled. What happens to your quotes, and
why in that direction?**

You're long, so the reservation price is already below the mid; doubling `σ` quadruples the
inventory-risk term, pushing the reservation price *further* below the mid and widening the
overall spread. Net effect: your ask drops (you're eager to sell and cut the long) and your
bid drops even more (you really don't want to buy more). Everything skews to shed the long,
harder than before, because that long is now four times as risky to hold.

**7. Why can't you backtest a market maker by replaying historical data the way you would a
directional strategy?**

Because your own quotes would have changed the book. In a replay you don't know your **queue
position**, so you can't know whether a trade that printed at your price would actually have
hit *your* order or someone ahead of you; and your resting size would have absorbed flow and
moved the very prices you're replaying. A directional strategy consumes liquidity at prices
that existed regardless of it; a market maker *provides* liquidity, so it's part of the
mechanism it's being tested against. That's *why* this project simulates rather than replays —
and being able to say that is worth more than any backtest number.

**8. What is queue position, and why does the model ignore it? What would change if it
didn't?**

At a given price level, orders fill in time priority — your place in that FIFO line is your
queue position. Being at the back means you fill *last*, i.e. only after enough volume trades
through, which is exactly when the level is about to be swept — you're filled precisely when
you least want to be. The Poisson fill model here ignores it: fills arrive as a pure rate
`λ(δ)` with no notion of a line. Adding queue position would make fills *more* adversely
selected (back-of-queue fills are the toxic ones) and would reward faster cancels and smaller
resting size — it would make naive look even worse.

**9. `κ` is high. What does that tell you about the market, and what does it do to your
spread?**

`κ` is how fast fill intensity decays as you back away from the mid. High `κ` means fills die
off quickly with distance — you only get filled very close to the mid, so the market is
tight/competitive and you have little room to charge for the spread. In the formula the
edge term `(2/γ)ln(1+γ/κ)` shrinks as `κ` rises, so high `κ` *compresses* your optimal spread:
quote wide and you simply won't trade.

**10. Where does this model break in real life?**

Everywhere the assumptions are clean and reality isn't: fills are Poisson with no queue and
no latency; the mid is arithmetic Brownian with no fat tails, no volatility clustering, no
jumps — and jumps, clustering, and fat tails are exactly what blow up real makers; there's no
market impact from your own quotes, no cancellations, no order-size distribution. And the
adverse selection here is injected as a first-order per-fill drift measured on *all* tape
trades, not the strategy's own fills. The model is a clean instrument for one idea — inventory
control under measured adverse selection — not a trading system.

**11. Is Avellaneda–Stoikov actually protecting you from adverse selection, or just from
inventory variance?**

Mostly the latter, and it's important to be honest about it. When I feed the measured
adverse selection back into the simulator, it costs the naive and the AS strategies a
*comparable* amount per fill (~−10% vs ~−7% of PnL) — AS does not dodge the adverse move.
What AS does is hold almost no inventory, which collapses the mark-to-market *variance* of
its PnL (several times smaller standard deviation than naive). That variance reduction is
the entire source of its much higher Sharpe. So the honest sentence is: *AS controls
inventory risk, which is a different and more defensible claim than "AS avoids adverse
selection."* How much of the mean-PnL hit it also avoids depends on how fast it can flatten
relative to how fast the adverse move arrives — which is a modelling choice the notebook
reports a sensitivity over rather than a single number.

**A note on getting the mid right — three tries.** The headline number here (how adverse
selection compares to the spread) took three attempts, and the repo shows all three.
(1) A first pass built the mid with a *centred* smoother that peeked one step into the
future — a lookahead bug that produced a tidy "naive loses ~2.5× the spread." (2) Fixing it
to a strictly *causal* trade-price proxy removed the lookahead, but a 1-second proxy still
cannot resolve BTC's true spread, so it honestly *declined to quote the ratio*. (3) Only the
real `bookTicker` quotes make the spread observable, and then the ratio (~20×) is defensible.
Owning that arc — including a lookahead bug of my own — is the part of this project I'd
actually want to talk about, more than any single number.
