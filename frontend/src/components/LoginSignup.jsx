import React, { useState } from "react";
import "../styles.css";

const LoginSignup = () => {
  const [tab, setTab] = useState("login");
  const [loginEmail, setLoginEmail] = useState("");
  const [loginPassword, setLoginPassword] = useState("");
  const [signupEmail, setSignupEmail] = useState("");
  const [signupPassword, setSignupPassword] = useState("");
  const [signupConfirm, setSignupConfirm] = useState("");

  return (
    <div className="card" style={{ maxWidth: 400, margin: "3rem auto 2rem auto", textAlign: "center" }}>
      <div style={{ display: "flex", justifyContent: "center", marginBottom: "2rem" }}>
        <button
          style={{
            background: tab === "login" ? "var(--accent)" : "#fff0fa",
            color: tab === "login" ? "#fff" : "#d72660",
            border: "none",
            borderRadius: "1.5rem 0 0 1.5rem",
            fontWeight: 700,
            fontSize: "1.1rem",
            padding: "0.8rem 2.2rem",
            cursor: "pointer",
            transition: "background 0.2s, color 0.2s"
          }}
          onClick={() => setTab("login")}
        >
          Login
        </button>
        <button
          style={{
            background: tab === "signup" ? "var(--accent)" : "#fff0fa",
            color: tab === "signup" ? "#fff" : "#d72660",
            border: "none",
            borderRadius: "0 1.5rem 1.5rem 0",
            fontWeight: 700,
            fontSize: "1.1rem",
            padding: "0.8rem 2.2rem",
            cursor: "pointer",
            transition: "background 0.2s, color 0.2s"
          }}
          onClick={() => setTab("signup")}
        >
          Sign Up
        </button>
      </div>
      {tab === "login" ? (
        <form>
          <label>Email</label>
          <input
            type="email"
            value={loginEmail}
            onChange={e => setLoginEmail(e.target.value)}
            placeholder="Enter your email"
            style={{ marginBottom: 16 }}
            required
          />
          <label>Password</label>
          <input
            type="password"
            value={loginPassword}
            onChange={e => setLoginPassword(e.target.value)}
            placeholder="Enter your password"
            required
          />
          <button type="submit" style={{ width: "100%", marginTop: 18 }}>Login</button>
        </form>
      ) : (
        <form>
          <label>Email</label>
          <input
            type="email"
            value={signupEmail}
            onChange={e => setSignupEmail(e.target.value)}
            placeholder="Enter your email"
            style={{ marginBottom: 16 }}
            required
          />
          <label>Password</label>
          <input
            type="password"
            value={signupPassword}
            onChange={e => setSignupPassword(e.target.value)}
            placeholder="Create a password"
            required
          />
          <label>Confirm Password</label>
          <input
            type="password"
            value={signupConfirm}
            onChange={e => setSignupConfirm(e.target.value)}
            placeholder="Confirm your password"
            required
          />
          <button type="submit" style={{ width: "100%", marginTop: 18 }}>Sign Up</button>
        </form>
      )}
    </div>
  );
};

export default LoginSignup;
